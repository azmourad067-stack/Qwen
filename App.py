# analyseur_hippique_geny.py
# -*- coding: utf-8 -*-
"""
Streamlit app — Extracteur Geny + Auto-training (hybride DL + XGBoost)
Usage: streamlit run analyseur_hippique_geny.py
"""

import os, re, warnings, math
from datetime import datetime
from itertools import combinations
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import joblib

warnings.filterwarnings("ignore")

# Optional ML imports
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, callbacks
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
except Exception:
    tf = None

from sklearn.preprocessing import StandardScaler

# Paths
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

HIST_PATH = os.path.join("data", "historique.csv")
DL_MODEL_PATH = os.path.join("models", "dl_model.keras")
SCALER_PATH = os.path.join("models", "scaler.joblib")
XGB_PATH = os.path.join("models", "xgb_model.joblib")
TRAIN_LOG = os.path.join("logs", "training_log.csv")

# ----------------- Utilities -----------------
def safe_float(x, default=np.nan):
    try:
        if pd.isna(x): return default
        s = str(x).strip().replace("\xa0"," ").replace(",", ".")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except:
        return default

def extract_weight(s):
    try:
        if pd.isna(s) or s=="":
            return 60.0
        m = re.search(r"(\d+(?:[.,]\d+)?)", str(s))
        return float(m.group(1).replace(",", ".")) if m else 60.0
    except:
        return 60.0

def clean_text(s):
    if pd.isna(s): return ""
    return re.sub(r"\s+", " ", str(s)).strip()

# ----------------- Geny-specific scraper -----------------
def scrape_geny_stats(url, timeout=12):
    """
    Scrape the given Geny stats URL (or similar). Try table first; otherwise parse the 'Ecarts et statistiques' text block.
    Returns a DataFrame with columns:
    ['Nom','Numéro de corde','Cote','Poids','Musique','Âge/Sexe','Jockey','Entraîneur','Gains']
    """
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    # Geny often uses ISO-8859-1; ensure encoding is correct
    if r.encoding is None or "utf" not in r.encoding.lower():
        r.encoding = "ISO-8859-1"
    soup = BeautifulSoup(r.text, "lxml")  # ensure lxml installed; fallback to html.parser if needed

    # 1) Try to find a direct table of partants with explicit markup
    # Common classes vary; try a few heuristics
    candidate_selectors = [
        lambda s: s.find("table", {"class":"table-partants"}),
        lambda s: s.find("table", {"id":"table-partants"}),
        lambda s: s.find("table")
    ]
    table = None
    for sel in candidate_selectors:
        try:
            table = sel(soup)
            if table is not None and len(table.find_all("tr"))>1:
                break
        except Exception:
            table = None

    rows = []
    if table is not None and len(table.find_all("tr"))>1:
        # parse rows robustly
        for tr in table.find_all("tr"):
            tds = tr.find_all(["td","th"])
            if not tds:
                continue
            texts = [td.get_text(" ", strip=True) for td in tds]
            # Skip header rows
            if len(texts)<2: continue
            # Heuristic: try to map columns
            # We'll search in texts for tokens that look like music (e.g. '1a2a3a' or 'Da' etc.) and for cote numbers
            name = ""
            num = ""
            cote = np.nan
            poids = ""
            musique = ""
            age_sexe = ""
            jockey = ""
            entraineur = ""
            gains = 0
            # try find name as the first text with letters and not only digits
            if len(texts)>=2:
                # often first is num second is name
                if re.match(r"^\d+$", texts[0].strip()):
                    num = texts[0].strip()
                    name = texts[1].strip()
                else:
                    # fallback: find the first text that contains letters and length > 1
                    for t in texts:
                        if re.search(r"[A-Za-zÀ-ÿ]", t):
                            name = t.strip()
                            break
            # find musique candidate: contains 'a' or 'Da' patterns
            for t in texts[::-1]:
                if re.search(r"\d+[aA]|Da|Dm|mDa|Da\d|[0-9]+a", t.replace(" ", "")):
                    musique = t.strip()
                    break
            # find cote candidate (number with comma/dot)
            for t in texts[::-1]:
                if re.search(r"\d+[,\.]\d+|\d+\s*\/\s*\d+|\d+$", t):
                    # attempt numeric draw-out
                    mf = re.search(r"\d+[,\.]\d+", t)
                    if mf:
                        cote = float(mf.group(0).replace(",", "."))
                        break
                    else:
                        # maybe integer represents gains etc, skip
                        pass
            # try gains near end: large numeric
            for t in texts[::-1]:
                digits = re.sub(r"[^\d]", "", t)
                if digits and len(digits) > 3:
                    try:
                        gains = int(digits)
                        break
                    except:
                        pass
            # poids and age might appear
            for t in texts:
                if re.search(r"\d{2}\,\d|\d{2}\.\d|\d{2}\s?kg|\d{2}\s?kg", t):
                    poids = t.strip()
                    break
                if re.search(r"\d+H|\d+M|\d+F", t):
                    age_sexe = t.strip()
                    break
            rows.append({
                "Nom": clean_text(name),
                "Numéro de corde": clean_text(num),
                "Cote": cote if not np.isnan(cote) else np.nan,
                "Poids": poids,
                "Musique": musique,
                "Âge/Sexe": age_sexe,
                "Jockey": jockey,
                "Entraîneur": entraineur,
                "Gains": gains
            })
    else:
        # 2) Fallback: parse textual 'Ecarts et statistiques' blocks
        text = soup.get_text("\n", strip=True)
        # isolate the block between 'Ecarts et statistiques' and 'Stats PMU mises à jour' or next header
        m = re.search(r"(Ecarts et statistiques.*?)(Stats PMU mises à jour|Protégé des antérieurs|$)", text, flags=re.I|re.S)
        block = m.group(1) if m else text[:4000]
        # split lines and find lines starting with number + name
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        for line in lines:
            # typical line example: "1 Incisif Vaucéen H7 2850 B. Rochard 1553 220 0 14% 1a 2a0m1a8mDa2a2m"
            if re.match(r"^\d+\s+\w+", line):
                parts = line.split()
                # take number
                num = parts[0]
                # attempt to find the name: from parts[1] until a token that looks like age (e.g. 'H7' or 'H8') or distance
                name_parts = []
                i = 1
                while i < len(parts):
                    if re.match(r"^[HMF]\d+|\d{3,4}$", parts[i]):  # age or distance token
                        break
                    name_parts.append(parts[i]); i += 1
                name = " ".join(name_parts)
                # search for musique pattern in the line
                mus = ""
                mus_m = re.search(r"([0-9aA]{1,3}a[0-9aA].+)$", line)
                if mus_m:
                    mus = mus_m.group(1)
                # search for jockey name (approx: initials and dot)
                jockey = ""
                j_m = re.search(r"([A-Z]\.\s?[A-Za-z-]+)", line)
                if j_m:
                    jockey = j_m.group(1)
                # gains: big numbers
                gains = 0
                g_m = re.search(r"(\d{4,})", line)
                if g_m:
                    gains = int(g_m.group(1))
                rows.append({
                    "Nom": clean_text(name),
                    "Numéro de corde": clean_text(num),
                    "Cote": np.nan,
                    "Poids": "",
                    "Musique": mus,
                    "Âge/Sexe": "",
                    "Jockey": jockey,
                    "Entraîneur": "",
                    "Gains": gains
                })
    if not rows:
        raise ValueError("Aucune donnée extraite (structure inattendue).")
    df = pd.DataFrame(rows)
    # post-cleaning: normalize names and fill defaults
    df["Nom"] = df["Nom"].fillna("").apply(lambda s: re.sub(r"[^\w\s'\-À-ÿ]", "", s))
    df["Cote"] = df["Cote"].apply(lambda x: safe_float(x, default=np.nan)).fillna(999)
    df["Poids"] = df["Poids"].apply(lambda x: extract_weight(x) if str(x).strip()!="" else 60.0)
    return df[["Nom","Numéro de corde","Cote","Poids","Musique","Âge/Sexe","Jockey","Entraîneur","Gains"]]

# ----------------- Feature engineering & prepare_data (same pipeline) -----------------
def music_to_features(music):
    s = str(music)
    digits = [int(x) for x in re.findall(r"\d+", s)]
    if not digits:
        return 0,0,0.0
    recent_wins = sum(1 for d in digits if d==1)
    recent_top3 = sum(1 for d in digits if d<=3)
    weights = np.linspace(1.0, 0.3, num=len(digits))
    weighted = sum((4-d)*w for d,w in zip(digits, weights)) / (len(digits)+1e-6)
    return recent_wins, recent_top3, weighted

def prepare_data(df):
    df = df.copy()
    for c in ["Nom","Numéro de corde","Cote","Poids","Musique","Âge/Sexe","Jockey","Entraîneur","Gains"]:
        if c not in df.columns:
            df[c] = ""
    df["odds_numeric"] = df["Cote"].apply(lambda x: safe_float(x, default=999)).fillna(999)
    df["draw_numeric"] = df["Numéro de corde"].apply(lambda x: safe_float(x, default=1)).fillna(1).astype(int)
    df["weight_kg"] = df["Poids"].apply(lambda x: extract_weight(x) if x!="" else 60.0)
    ages=[]; is_f=[]; rw=[]; r3=[]; rwght=[]
    for a in df.get("Âge/Sexe", [""]*len(df)):
        m = re.search(r"(\d+)", str(a))
        ages.append(float(m.group(1)) if m else 4.0)
        s = str(a).upper()
        is_f.append(1 if "F" in s else 0)
    for m in df.get("Musique", [""]*len(df)):
        a,b,c = music_to_features(m)
        rw.append(a); r3.append(b); rwght.append(c)
    df["age"] = ages
    df["is_female"] = is_f
    df["recent_wins"] = rw
    df["recent_top3"] = r3
    df["recent_weighted"] = rwght
    df = df[df["odds_numeric"] > 0]
    df = df.reset_index(drop=True)
    return df

# ----------------- Hybrid model manager (DL + XGB fallback) -----------------
class HybridModel:
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or ["odds_numeric","draw_numeric","weight_kg","age","is_female","recent_wins","recent_top3","recent_weighted"]
        self.scaler = StandardScaler()
        self.dl_model = None
        self.xgb_model = None
        # try load persisted
        try:
            if tf is not None and os.path.exists(DL_MODEL_PATH):
                self.dl_model = tf.keras.models.load_model(DL_MODEL_PATH)
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
            if xgb is not None and os.path.exists(XGB_PATH):
                self.xgb_model = joblib.load(XGB_PATH)
        except Exception as e:
            st.warning(f"Chargement modèle existant échoué: {e}")

    def build_dl(self, input_dim):
        if tf is None:
            return None
        m = Sequential([
            Dense(128, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="linear")
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    def train(self, X, y, dl_epochs=16, dl_batch=8, xgb_rounds=100, val_split=0.15, use_cv=False):
        X_np = X.values if hasattr(X, "values") else np.array(X)
        y_np = y.values if hasattr(y, "values") else np.array(y)
        Xs = self.scaler.fit_transform(X_np)
        hist = None
        if tf is not None:
            if self.dl_model is None:
                self.dl_model = self.build_dl(Xs.shape[1])
            es = callbacks.EarlyStopping(patience=6, restore_best_weights=True)
            hist = self.dl_model.fit(Xs, y_np, validation_split=val_split, epochs=dl_epochs, batch_size=dl_batch, callbacks=[es], verbose=0)
            try:
                self.dl_model.save(DL_MODEL_PATH, overwrite=True)
            except Exception as e:
                st.warning(f"Impossible sauvegarder modèle DL: {e}")
        if xgb is not None:
            try:
                if use_cv and len(Xs) >= 50:
                    dtrain = xgb.DMatrix(Xs, label=y_np)
                    params = {"objective":"reg:squarederror","learning_rate":0.05,"max_depth":4}
                    cvres = xgb.cv(params, dtrain, num_boost_round=xgb_rounds, nfold=4, early_stopping_rounds=10, verbose_eval=False)
                    best_rounds = len(cvres)
                else:
                    best_rounds = max(10, min(200, xgb_rounds))
                self.xgb_model = xgb.XGBRegressor(n_estimators=best_rounds, learning_rate=0.05, max_depth=4, random_state=42)
                self.xgb_model.fit(Xs, y_np, verbose=False)
                joblib.dump(self.xgb_model, XGB_PATH)
            except Exception as e:
                st.warning(f"Erreur XGBoost training: {e}")
                self.xgb_model = None
        try:
            joblib.dump(self.scaler, SCALER_PATH)
        except Exception as e:
            st.warning(f"Erreur sauvegarde scaler: {e}")
        loss = None
        if hist is not None:
            loss = hist.history.get("loss", [None])[-1]
        with open(TRAIN_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()},{len(Xs)},{loss},{DL_MODEL_PATH}\n")
        return hist.history if hist is not None else None

    def predict(self, X_df):
        if len(X_df)==0:
            return np.array([]), np.array([]), np.array([])
        X_np = X_df.values if hasattr(X_df, "values") else np.array(X_df)
        Xs = self.scaler.transform(X_np)
        dl_preds = np.zeros(len(Xs)); xgb_preds = np.zeros(len(Xs))
        if self.dl_model is not None and tf is not None:
            try:
                dl_preds = self.dl_model.predict(Xs).flatten()
            except:
                dl_preds = np.zeros(len(Xs))
        if self.xgb_model is not None:
            try:
                xgb_preds = self.xgb_model.predict(Xs)
            except:
                xgb_preds = np.zeros(len(Xs))
        def norm(a):
            a = np.array(a, dtype=float)
            if a.size==0 or a.max()==a.min():
                return np.zeros_like(a)
            return (a - a.min())/(a.max()-a.min())
        ndl = norm(dl_preds); nx = norm(xgb_preds)
        ensemble = 0.6*ndl + 0.4*nx if (nx.size and ndl.size) else (ndl if ndl.size else nx)
        return ensemble, ndl, nx

# ----------------- Helpers: confidence & combos -----------------
def compute_confidence(a,b):
    if a.size==0: return np.zeros_like(b)
    if b.size==0: return np.ones_like(a)
    arr = np.vstack([a,b])
    std = np.std(arr, axis=0)
    conf = 1.0 - (std / (std.max()+1e-9))
    return np.clip(conf, 0.0, 1.0)

def generate_weighted_trios(df_ranked, n=35, max_total_odds=120.0, avoid_same_trainer=True):
    names = df_ranked["Nom"].tolist()
    if len(names)<3: return []
    scores = df_ranked["score_final"].values
    scores = np.maximum(scores, 1e-9)
    probs = scores / scores.sum()
    combos_set=set(); results=[]
    attempts=0; max_attempts=max(2000, n*200)
    while len(results)<n and attempts<max_attempts:
        attempts+=1
        chosen=list(np.random.choice(names,size=3,replace=False,p=probs))
        key=tuple(sorted(chosen))
        if key in combos_set: continue
        sub=df_ranked[df_ranked["Nom"].isin(chosen)]
        try:
            total_odds = sub["odds_numeric"].astype(float).sum()
            if total_odds>max_total_odds: continue
        except: pass
        if avoid_same_trainer and "Entraîneur" in sub.columns:
            trainers = sub["Entraîneur"].astype(str).tolist()
            if len(set(trainers))<len(trainers): continue
        combos_set.add(key); results.append(tuple(chosen))
    return results

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="🏇 Analyseur Hippique Geny", layout="wide", page_icon="🏇")
st.title("🏇 Analyseur Hippique — Geny Scraper + Auto-training Hybrid")

with st.sidebar:
    st.header("Configuration")
    url_input = st.text_input("URL Geny (stats/partants) example:", value="https://www.geny.com/stats-pmu?id_course=1610442&info=2025-10-24-Vincennes-pmu-Prix+Orionis")
    use_scraper = st.checkbox("Utiliser scraper Geny", value=True)
    auto_train = st.checkbox("Auto-train (historique)", value=True)
    dl_epochs = st.number_input("DL epochs", min_value=2, max_value=400, value=16, step=2)
    dl_batch = st.number_input("DL batch", min_value=2, max_value=64, value=8, step=1)
    use_cv = st.checkbox("Use XGBoost CV if historique large", value=True)
    blend_weight = st.slider("Poids modèles (1=100% modèles, 0=heuristique)", 0.0, 1.0, 0.6)
    n_combos = st.number_input("Nb combinaisons e-trio", min_value=5, max_value=200, value=35)

tab1, tab2 = st.tabs(["Scraping & Load","Historique / Models"])
df_race = None

with tab1:
    st.subheader("Scraper Geny")
    st.markdown("Colle l'URL Geny et clique 'Charger la course'")
    if st.button("Charger la course depuis Geny"):
        if not url_input:
            st.error("Fournis une URL")
        else:
            try:
                df_race = scrape_geny_stats(url_input)
                st.success(f"Extraction OK ({len(df_race)} lignes)")
                st.dataframe(df_race)
            except Exception as e:
                st.error(f"Erreur scraping: {e}")

with tab2:
    st.subheader("Historique et modèles")
    if os.path.exists(HIST_PATH):
        hist = pd.read_csv(HIST_PATH)
        st.markdown(f"Historique total: {len(hist)} lignes")
        if st.button("Afficher dernières lignes de l'historique"):
            st.dataframe(hist.tail(50))
    else:
        st.info("Aucun historique trouvé pour le moment.")

# main pipeline: accept uploaded CSV as well
uploaded = st.file_uploader("Ou charge un CSV local (Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe, Jockey, Entraîneur, Gains)", type=["csv"])
if uploaded is not None:
    try:
        df_race = pd.read_csv(uploaded)
        st.success(f"CSV chargé ({len(df_race)} lignes)")
        st.dataframe(df_race.head())
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}")

if df_race is not None and len(df_race)>0:
    st.markdown("---")
    st.header("Préparation & Auto-entrainement")
    df_prep = prepare_data(df_race)
    st.dataframe(df_prep[["Nom","Cote","odds_numeric","draw_numeric","weight_kg","recent_wins","recent_top3","recent_weighted"]])

    # append to historique
    try:
        df_raw = df_race.copy()
        df_raw["_source_ts"] = datetime.now().isoformat()
        if os.path.exists(HIST_PATH):
            old = pd.read_csv(HIST_PATH)
            combined = pd.concat([old, df_raw], ignore_index=True)
        else:
            combined = df_raw
        combined.to_csv(HIST_PATH, index=False)
        st.success(f"Historique mis à jour ({len(combined)} lignes totales).")
    except Exception as e:
        st.warning(f"Impossible mettre à jour historique: {e}")

    # build and train models
    feats = ["odds_numeric","draw_numeric","weight_kg","age","is_female","recent_wins","recent_top3","recent_weighted"]
    manager = HybridModel(feature_cols=feats)
    # try prepare full historique for training if exists
    X_hist=None; y_hist=None
    if os.path.exists(HIST_PATH):
        try:
            hist_raw = pd.read_csv(HIST_PATH)
            hist_prep = prepare_data(hist_raw)
            # if historic placements exist, use them; else pseudo-target
            if "placement" in hist_raw.columns or "rank" in hist_raw.columns:
                if "placement" in hist_raw.columns:
                    y_hist = 1.0/(hist_prep["placement"].astype(float)+0.1)
                else:
                    y_hist = 1.0/(hist_prep["rank"].astype(float)+0.1)
                X_hist = hist_prep[feats]
            else:
                y_hist = 0.7*(1.0/(hist_prep["odds_numeric"]+0.1)) + 0.3*(hist_prep["recent_weighted"]/(hist_prep["recent_weighted"].max()+1e-6))
                X_hist = hist_prep[feats]
        except Exception as e:
            st.warning(f"Erreur lecture historique: {e}")

    trained = False
    if auto_train:
        try:
            if X_hist is not None and len(X_hist) >= 4:
                st.info(f"Entraînement automatique sur historique ({len(X_hist)} échantillons)...")
                manager.train(X_hist, y_hist, dl_epochs=int(dl_epochs), dl_batch=int(dl_batch), use_cv=use_cv)
                trained = True
            else:
                if len(df_prep) >= 3:
                    y_curr = 0.7*(1.0/(df_prep["odds_numeric"]+0.1)) + 0.3*(df_prep["recent_weighted"]/(df_prep["recent_weighted"].max()+1e-6))
                    st.info("Peu de données historiques: entraînement sur la course actuelle (pseudo-target).")
                    manager.train(df_prep[feats], y_curr, dl_epochs=max(4,int(dl_epochs//2)), dl_batch=int(dl_batch), use_cv=False)
                    trained = True
        except Exception as e:
            st.warning(f"Erreur auto-train: {e}")

    # predictions & final scoring
    X_curr = df_prep[feats].fillna(0)
    ensemble, dl_norm, xgb_norm = manager.predict(X_curr)
    heuristic = 1.0/(df_prep["odds_numeric"]+0.1)
    if heuristic.max() != heuristic.min():
        heuristic = (heuristic - heuristic.min())/(heuristic.max()-heuristic.min())
    final = (1 - blend_weight) * heuristic + blend_weight * ensemble
    conf = compute_confidence(dl_norm, xgb_norm)
    df_prep["dl_norm"] = dl_norm
    df_prep["xgb_norm"] = xgb_norm if len(xgb_norm)==len(dl_norm) else np.zeros_like(dl_norm)
    df_prep["score_final"] = final
    df_prep["confidence"] = conf
    df_ranked = df_prep.sort_values("score_final", ascending=False).reset_index(drop=True)
    df_ranked["rang"] = range(1, len(df_ranked)+1)

    # display
    left, right = st.columns([2,1])
    with left:
        st.subheader("Classement final")
        disp = df_ranked[["rang","Nom","Cote","Numéro de corde","weight_kg","score_final","confidence"]].copy()
        disp["Score"] = disp["score_final"].round(3)
        disp["Conf"] = (disp["confidence"]*100).round(1).astype(str) + "%"
        disp = disp.drop(["score_final","confidence"], axis=1)
        st.dataframe(disp, use_container_width=True)
    with right:
        st.subheader("Infos & metrics")
        st.metric("Nb chevaux", len(df_ranked))
        st.markdown(f"- Auto-train: **{'ON' if auto_train else 'OFF'}**")
        st.markdown(f"- Modèles DL: {'Oui' if manager.dl_model is not None else 'Non'}, XGB: {'Oui' if manager.xgb_model is not None else 'Non'}")
        if trained:
            st.success("✅ Auto-train effectué")

    st.subheader("Visualisations")
    st.bar_chart(df_ranked.set_index("Nom")["score_final"].head(12))

    st.subheader("Générateur e-trio pondéré & filtré")
    combos = generate_weighted_trios(df_ranked, n=int(n_combos))
    if combos:
        for i,c in enumerate(combos):
            st.markdown(f"{i+1}. {c[0]} — {c[1]} — {c[2]}")
    else:
        st.info("Aucune combinaison générée")

    st.subheader("Export & snapshot")
    st.download_button("Télécharger pronostic CSV", df_ranked.to_csv(index=False), f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    st.download_button("Télécharger pronostic JSON", df_ranked.to_json(orient="records", indent=2), f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    snapshot = os.path.join("data", f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_ranked.to_csv(snapshot, index=False)
    st.success(f"Snapshot sauvegardé: {snapshot}")

st.markdown("---")
st.info("Conseils: pour des pronostics fiables, alimente l'historique avec des résultats labellisés (placements). Je peux automatiser la récupération des résultats si tu veux.")
