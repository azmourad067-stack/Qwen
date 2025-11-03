"""
🏇 Analyseur Hippique IA Pro — Deep Learning + Geny Scraper v6
Auteur : GPT-5
Version stable avec sécurité anti-doublons et Deep Learning Keras
Compatible : Streamlit / Pydroid
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, time, os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px

# ==============================
# CONFIGS
# ==============================
HEADERS = {"User-Agent": "Mozilla/5.0"}
SCRAPE_DEBUG_DIR = "scrape_debug"
os.makedirs(SCRAPE_DEBUG_DIR, exist_ok=True)

def safe_float(x, default=np.nan):
    try:
        return float(str(x).replace(",", "."))
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str): return np.nan
    m = re.search(r"(\d+(?:[.,]\d+)?)", str(poids_str))
    return float(m.group(1).replace(",", ".")) if m else np.nan

def extract_music_features(music):
    if pd.isna(music) or str(music).strip() == "":
        return {"wins":0,"places":0,"total":0,"win_rate":0,"place_rate":0,"recent_form":0}
    s = str(music)
    pos = [int(d) for d in re.findall(r"(\d+)", s) if int(d)>0]
    if len(pos)==0:
        return {"wins":0,"places":0,"total":0,"win_rate":0,"place_rate":0,"recent_form":0}
    total=len(pos)
    wins=sum(1 for p in pos if p==1)
    places=sum(1 for p in pos if p<=3)
    recent=pos[:3]
    recent_form=sum(1/p for p in recent)/len(recent)
    return {"wins":wins,"places":places,"total":total,
            "win_rate":wins/total,"place_rate":places/total,"recent_form":recent_form}

def save_debug_html(url, html):
    fn=os.path.join(SCRAPE_DEBUG_DIR,f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(fn,"w",encoding="utf-8") as f:
        f.write(f"<!-- URL: {url} -->\n{html}")
    return fn

def get_html(url,timeout=10):
    try:
        r=requests.get(url,headers=HEADERS,timeout=timeout)
        if r.status_code==200: return r.text,None
        return None,f"HTTP {r.status_code}"
    except Exception as e:
        return None,str(e)

def parse_table_by_headers(soup,wanted_headers):
    tables=soup.find_all("table")
    for tbl in tables:
        ths=[th.get_text(strip=True).lower() for th in tbl.find_all(["th","td"])[:10]]
        if any(any(h in th for th in ths) for h in wanted_headers):
            return tbl
    return None

def scrape_geny_partants(url):
    html,err=get_html(url)
    if html is None: return None,err
    path=save_debug_html(url,html)
    soup=BeautifulSoup(html,"html.parser")
    wanted=["nom","cote","poids","musique","num"]
    tbl=parse_table_by_headers(soup,wanted)
    if not tbl: return None,f"⚠️ Table non trouvée (debug: {path})"
    rows=[]
    for tr in tbl.find_all("tr"):
        cols=[td.get_text(" ",strip=True) for td in tr.find_all(["td","th"])]
        if len(cols)>2: rows.append(cols)
    if len(rows)<2: return None,f"Table vide (debug: {path})"
    hdr=[h.lower() for h in rows[0]]
    def find(names):
        for n in names:
            for i,h in enumerate(hdr):
                if n in h: return i
        return None
    idx_nom=find(["nom"]); idx_cote=find(["cote"])
    idx_poids=find(["poids"]); idx_mus=find(["musique"])
    idx_num=find(["num","corde"])
    parsed=[]
    for r in rows[1:]:
        def sg(i): return r[i] if (i is not None and i<len(r)) else ""
        parsed.append({
            "Num":sg(idx_num),"Nom":sg(idx_nom),"Cote":sg(idx_cote),
            "Poids":sg(idx_poids),"Musique":sg(idx_mus)
        })
    return pd.DataFrame(parsed),f"✅ Partants extraits (debug: {path})"

def scrape_geny_stats(url):
    html,err=get_html(url)
    if html is None: return None,err
    path=save_debug_html(url,html)
    soup=BeautifulSoup(html,"html.parser")
    wanted=["nom","arriv","position"]
    tbl=parse_table_by_headers(soup,wanted)
    if not tbl: return None,f"⚠️ Table non trouvée (debug: {path})"
    rows=[]
    for tr in tbl.find_all("tr"):
        cols=[td.get_text(" ",strip=True) for td in tr.find_all(["td","th"])]
        if len(cols)>2: rows.append(cols)
    if len(rows)<2: return None,f"Table vide (debug: {path})"
    hdr=[h.lower() for h in rows[0]]
    def find(names):
        for n in names:
            for i,h in enumerate(hdr):
                if n in h: return i
        return None
    idx_nom=find(["nom"]); idx_pos=find(["arriv","position"])
    parsed=[]
    for r in rows[1:]:
        def sg(i): return r[i] if (i is not None and i<len(r)) else ""
        parsed.append({"Nom":sg(idx_nom),"Position":sg(idx_pos)})
    return pd.DataFrame(parsed),f"✅ Stats extraites (debug: {path})"

# ==============================
# INTERFACE STREAMLIT
# ==============================
st.set_page_config(page_title="Analyseur Hippique IA Deep v6", layout="wide")
st.title("🏇 Analyseur Hippique IA Pro — Deep Learning + Geny Scraper v6")

col1,col2=st.columns(2)
url_partants=col1.text_input("URL Partants (Geny)")
url_stats=col2.text_input("URL Stats (Geny)")
if st.button("🔍 Extraire et Analyser"):
    dfp,msg1=scrape_geny_partants(url_partants)
    st.info(msg1)
    dfs,msg2=scrape_geny_stats(url_stats)
    st.info(msg2)

    if dfp is None:
        st.error("❌ Impossible d'extraire les partants.")
        st.stop()

    st.subheader("✅ Partants extraits")
    st.dataframe(dfp, use_container_width=True)

    # --- Préparation Features ---
    dfp["Nom"]=dfp["Nom"].str.strip().str.upper()
    dfp["odds_numeric"]=dfp["Cote"].apply(lambda x:safe_float(x,999))
    dfp["weight_kg"]=dfp["Poids"].apply(extract_weight)
    dfp["odds_inv"]=1/(dfp["odds_numeric"]+0.1)
    mus_feats=dfp["Musique"].apply(extract_music_features).apply(pd.Series)
    dfp=pd.concat([dfp,mus_feats],axis=1)

    # --- Historique ---
    dfp["historical_podium_rate"]=0.0
    if dfs is not None and len(dfs)>0 and "Position" in dfs.columns:
        dfs["Nom"]=dfs["Nom"].str.strip().str.upper()
        dfs["Position_num"]=dfs["Position"].apply(lambda x:safe_float(x,np.nan))
        valid=dfs[dfs["Position_num"].notna()]
        if len(valid)>0:
            hist=valid.groupby("Nom").apply(lambda g:(g["Position_num"]<=3).mean()).reset_index()
            hist.columns=["Nom","historical_podium_rate"]
            dfp=dfp.merge(hist,on="Nom",how="left")
            dfp["historical_podium_rate"]=dfp["historical_podium_rate"].fillna(0.0)

    # === Features finales ===
    features=["odds_inv","win_rate","place_rate","recent_form","weight_kg","historical_podium_rate"]
    X=dfp[features].fillna(0.0).values

    # === Deep Learning Model ===
    scaler=StandardScaler()
    Xs=scaler.fit_transform(X)
    model=Sequential([
        Dense(64,activation='relu',input_shape=(Xs.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32,activation='relu'),
        Dropout(0.3),
        Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    y=(dfp["odds_numeric"]<10).astype(int)  # pseudo-cible : chevaux favoris
    es=EarlyStopping(monitor='loss',patience=5,restore_best_weights=True)
    model.fit(Xs,y,epochs=40,batch_size=4,verbose=0,callbacks=[es])
    preds=model.predict(Xs).flatten()
    dfp["score_pred"]=preds
    dfp["score_pred_norm"]=(dfp["score_pred"]-dfp["score_pred"].min())/(dfp["score_pred"].max()-dfp["score_pred"].min())

    # --- Anti-doublons avant affichage ---
    if dfp.columns.duplicated().any():
        st.warning(f"⚠️ Colonnes dupliquées supprimées : {dfp.columns[dfp.columns.duplicated()].tolist()}")
        dfp=dfp.loc[:,~dfp.columns.duplicated()]

    st.subheader("🏁 Classement Deep Learning")
    df_sorted=dfp.sort_values("score_pred_norm",ascending=False)
    st.dataframe(df_sorted[["Nom","Cote","score_pred_norm","historical_podium_rate"]], use_container_width=True)

    fig=px.bar(df_sorted,x="Nom",y="score_pred_norm",color="historical_podium_rate",
               color_continuous_scale="Viridis",title="Scores Deep Learning (probabilité relative de podium)")
    st.plotly_chart(fig,use_container_width=True)

    csv=df_sorted.to_csv(index=False)
    st.download_button("📄 Télécharger Résultats CSV",csv,
                       file_name=f"deep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
