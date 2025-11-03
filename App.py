"""
app_geny_dl.py
🏇 Analyseur Hippique IA Pro vDL — Scraping Geny + Deep Learning + Calibration
Usage: streamlit run app_geny_dl.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import time
import json
from datetime import datetime
from io import BytesIO

import requests
from bs4 import BeautifulSoup

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import plotly.express as px

# ---------------------------
# Configs & paths
# ---------------------------
MODEL_PATH = "hippo_model.h5"
SCALER_PATH = "hippo_scaler.joblib"
CALIB_PATH = "hippo_calibrator.joblib"
SCRAPE_DEBUG_DIR = "scrape_debug"
os.makedirs(SCRAPE_DEBUG_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ---------------------------
# Utilities
# ---------------------------
def safe_float(x, default=np.nan):
    try:
        if pd.isna(x): return default
        return float(str(x).replace(',', '.'))
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str) or str(poids_str).strip() == "":
        return np.nan
    m = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(m.group(1).replace(',', '.')) if m else np.nan

def extract_music_features(music):
    """
    Parse music strings like '1a2a3a1a' or '1-2-3' into summary features.
    Returns dict with wins, places, total_races, win_rate, place_rate, recent_form, avg_pos.
    """
    if pd.isna(music) or str(music).strip() == "":
        return {'wins':0,'places':0,'total_races':0,'win_rate':0.0,'place_rate':0.0,'recent_form':0.0,'avg_pos':np.nan}
    s = str(music)
    positions = [int(x) for x in re.findall(r'(\d+)', s) if int(x) > 0]
    if len(positions) == 0:
        return {'wins':0,'places':0,'total_races':0,'win_rate':0.0,'place_rate':0.0,'recent_form':0.0,'avg_pos':np.nan}
    total = len(positions)
    wins = sum(1 for p in positions if p == 1)
    places = sum(1 for p in positions if p <= 3)
    recent = positions[:3]
    recent_form = sum(1.0/p for p in recent)/len(recent) if len(recent)>0 else 0.0
    avg_pos = float(np.mean(positions))
    return {'wins':wins,'places':places,'total_races':total,'win_rate':wins/total,'place_rate':places/total,'recent_form':recent_form,'avg_pos':avg_pos}

def save_debug_html(url_or_tag, html):
    fn = os.path.join(SCRAPE_DEBUG_DIR, f"debug_{url_or_tag.replace('/','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(f"<!-- source: {url_or_tag} -->\n")
        f.write(html)
    return fn

def get_html(url, timeout=12):
    """Download HTML with polite wait and UA; return (html, error)"""
    try:
        time.sleep(0.6)
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r.text, None
        else:
            return None, f"HTTP {r.status_code}"
    except Exception as e:
        return None, str(e)

def parse_table_by_headers(soup, wanted_headers):
    """
    Find a table where first row / header contains some of wanted_headers
    Returns the table element or None.
    """
    tables = soup.find_all("table")
    for tbl in tables:
        # get first 12 header-like cells text
        ths = [th.get_text(strip=True).lower() for th in tbl.find_all(["th","td"])[:12]]
        if any(any(h in th for th in ths) for h in wanted_headers):
            return tbl
    return None

# ---------------------------
# Scrapers (Geny-focused heuristics)
# ---------------------------
def scrape_geny_partants(url):
    html, err = get_html(url)
    if html is None:
        return None, f"Erreur téléchargement partants: {err}"
    debug_path = save_debug_html("partants", html)
    soup = BeautifulSoup(html, "html.parser")
    wanted = ['nom', 'cote', 'poids', 'musique', 'num', 'corde']
    tbl = parse_table_by_headers(soup, wanted)
    if tbl:
        rows = []
        for tr in tbl.find_all("tr"):
            cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
            if len(cols) >= 2:
                rows.append(cols)
        if len(rows) < 2:
            return None, f"Table trouvée mais vide (debug: {debug_path})"
        header = [h.lower() for h in rows[0]]
        def idx_for(names):
            for n in names:
                for i, h in enumerate(header):
                    if n in h:
                        return i
            return None
        i_num = idx_for(['num','n°','corde'])
        i_nom = idx_for(['nom','name'])
        i_cote = idx_for(['cote'])
        i_poids = idx_for(['poids','weight'])
        i_mus = idx_for(['musique','music'])
        parsed = []
        for r in rows[1:]:
            def sg(i):
                return r[i] if (i is not None and i < len(r)) else ''
            parsed.append({
                'Numéro de corde': sg(i_num),
                'Nom': sg(i_nom),
                'Cote': sg(i_cote),
                'Poids': sg(i_poids),
                'Musique': sg(i_mus)
            })
        return pd.DataFrame(parsed), f"Partants extraits depuis tableau HTML. Debug: {debug_path}"
    # fallback: try extracting JSON in scripts
    scripts = soup.find_all('script')
    for sc in scripts:
        if sc.string and ('partants' in sc.string.lower() or 'participants' in sc.string.lower()):
            txt = sc.string
            try:
                # try to find json object
                jms = re.findall(r'(\{.*\}|\[.*\])', txt, flags=re.S)
                for jm in jms:
                    try:
                        data = json.loads(jm)
                        # find list of dicts
                        if isinstance(data, dict):
                            for k,v in data.items():
                                if isinstance(v, list) and len(v)>0 and isinstance(v[0], dict):
                                    if any(key in v[0].keys() for key in ['name','nom','cote','poids']):
                                        df = pd.DataFrame(v)
                                        return df, "Partants extraits depuis JSON dans script"
                    except:
                        continue
            except:
                continue
    return None, f"Aucun partant structuré trouvé (debug: {debug_path})"

def scrape_geny_stats(url):
    html, err = get_html(url)
    if html is None:
        return None, f"Erreur téléchargement stats: {err}"
    debug_path = save_debug_html("stats", html)
    soup = BeautifulSoup(html, "html.parser")
    wanted = ['date','hippodrome','nom','position','arriv','résultat','cote']
    tbl = parse_table_by_headers(soup, wanted)
    if tbl:
        rows = []
        for tr in tbl.find_all("tr"):
            cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
            if len(cols) >= 2:
                rows.append(cols)
        if len(rows) < 2:
            return None, f"Table stats trouvée mais vide (debug: {debug_path})"
        header = [h.lower() for h in rows[0]]
        def idx_for(names):
            for n in names:
                for i,h in enumerate(header):
                    if n in h:
                        return i
            return None
        i_date = idx_for(['date'])
        i_nom = idx_for(['nom','horse'])
        i_pos = idx_for(['arriv','position','résultat','resultat','rang'])
        i_cote = idx_for(['cote'])
        parsed = []
        for r in rows[1:]:
            def sg(i): return r[i] if (i is not None and i < len(r)) else ''
            parsed.append({
                'Date': sg(i_date),
                'Nom': sg(i_nom),
                'Position': sg(i_pos),
                'Cote': sg(i_cote)
            })
        return pd.DataFrame(parsed), f"Stats extraites depuis tableau HTML. Debug: {debug_path}"
    # fallback: extract lines with dates heuristically
    text_lines = soup.get_text("\n", strip=True).split("\n")
    candidates = [l for l in text_lines if re.search(r'\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}', l)]
    if candidates:
        parsed = []
        for l in candidates[:300]:
            date_m = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', l)
            date = date_m.group(1) if date_m else ''
            # try find position inside text (single digit)
            pos_m = re.search(r'\b([1-9]|1[0-9])\b', l)
            pos = pos_m.group(1) if pos_m else ''
            parsed.append({'Date': date, 'Nom': l[:40], 'Position': pos, 'Cote': ''})
        return pd.DataFrame(parsed), f"Stats heuristiques extraites du texte (debug: {debug_path})"
    return None, f"Aucun historique trouvé (debug: {debug_path})"

# ---------------------------
# Feature preparation (FIX: éviter les doublons de colonnes)
# ---------------------------
def prepare_features_from_partants(dfp, stats_df=None):
    """
    Input: dfp (DataFrame of partants), optional stats DataFrame (historical)
    Output: prepared DataFrame with numeric features & cleaned names
    """
    df = dfp.copy()
    
    # Normalize name
    if 'Nom' in df.columns:
        df['Nom'] = df['Nom'].astype(str).str.strip()
    else:
        df['Nom'] = df.index.astype(str)

    # numeric conversions
    if 'Cote' in df.columns:
        df['odds_numeric'] = df['Cote'].apply(lambda x: safe_float(x, default=np.nan))
        median_odds = np.nanmedian(df['odds_numeric'].values)
        if np.isnan(median_odds):
            median_odds = 5.0
        df['odds_numeric'] = df['odds_numeric'].fillna(median_odds)
    else:
        df['odds_numeric'] = 5.0

    df['odds_inv'] = 1.0 / (df['odds_numeric'] + 0.1)

    if 'Poids' in df.columns:
        df['weight_kg'] = df['Poids'].apply(lambda x: extract_weight(x)).fillna(df['Poids'].apply(lambda x: safe_float(x, np.nan)).fillna(60.0))
    else:
        df['weight_kg'] = 60.0

    # draw / number
    if 'Numéro de corde' in df.columns:
        df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: int(re.sub(r'\D','',str(x))) if re.sub(r'\D','',str(x)) else 1)
    elif 'Num' in df.columns:
        df['draw_numeric'] = df['Num'].apply(lambda x: int(re.sub(r'\D','',str(x))) if re.sub(r'\D','',str(x)) else 1)
    else:
        df['draw_numeric'] = np.arange(1, len(df)+1)

    # music extraction
    if 'Musique' in df.columns:
        music_feats = df['Musique'].apply(extract_music_features).apply(pd.Series)
        # Renommer pour éviter conflit si colonnes existent déjà
        for col in music_feats.columns:
            if col in df.columns:
                df = df.drop(columns=[col])
        df = pd.concat([df.reset_index(drop=True), music_feats.reset_index(drop=True)], axis=1)
    else:
        # fill defaults
        if 'wins' not in df.columns:
            df['wins'] = 0
        if 'places' not in df.columns:
            df['places'] = 0
        if 'total_races' not in df.columns:
            df['total_races'] = 0
        if 'win_rate' not in df.columns:
            df['win_rate'] = 0.0
        if 'place_rate' not in df.columns:
            df['place_rate'] = 0.0
        if 'recent_form' not in df.columns:
            df['recent_form'] = 0.0
        if 'avg_pos' not in df.columns:
            df['avg_pos'] = np.nan

    # parse Age if present
    if 'Age' in df.columns:
        df['age_num'] = df['Age'].apply(lambda x: safe_float(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else 4.0)
    elif 'Âge/Sexe' in df.columns:
        df['age_num'] = df['Âge/Sexe'].str.extract(r'(\d+)').astype(float).fillna(4.0).astype(float)
    else:
        df['age_num'] = 4.0

    # historical podium rate from stats_df
    df['historical_podium_rate'] = 0.0
    if stats_df is not None and not stats_df.empty:
        stats = stats_df.copy()
        if 'Nom' in stats.columns and 'Position' in stats.columns:
            # normalize names
            stats['Nom'] = stats['Nom'].astype(str).str.strip()
            # numeric position
            stats['Position_num'] = stats['Position'].apply(lambda x: safe_float(x, default=np.nan))
            stats_valid = stats[stats['Position_num'].notna()].copy()
            if len(stats_valid) > 0:
                hist = stats_valid.groupby('Nom').agg(
                    hist_total = ('Position_num','count'),
                    hist_wins = ('Position_num', lambda s: int((s==1).sum()))
                ).reset_index()
                podiums = stats_valid.groupby('Nom').apply(lambda g: int((g['Position_num']<=3).sum())).rename('hist_podiums').reset_index()
                hist = hist.merge(podiums, on='Nom', how='left')
                hist['hist_podium_rate'] = hist.apply(lambda r: (r['hist_podiums'] / r['hist_total']) if r['hist_total']>0 else 0.0, axis=1)
                podium_map = dict(zip(hist['Nom'], hist['hist_podium_rate']))
                df['historical_podium_rate'] = df['Nom'].map(podium_map).fillna(0.0)

    # Définir les features finales SANS créer de doublons
    feature_cols = ['odds_inv', 'win_rate', 'place_rate', 'recent_form', 'weight_kg', 'age_num', 'historical_podium_rate', 'draw_numeric']
    
    # S'assurer que toutes les features existent
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    
    # Remplir les NaN
    for col in feature_cols:
        df[col] = df[col].fillna(0.0)
    
    return df, feature_cols

# ---------------------------
# Model building & training
# ---------------------------
def build_keras_model(input_dim, units1=64, units2=32, dropout=0.2, lr=1e-3):
    tf.keras.backend.clear_session()
    m = Sequential()
    m.add(Dense(units1, activation='relu', input_shape=(input_dim,)))
    m.add(BatchNormalization())
    if dropout > 0:
        m.add(Dropout(dropout))
    m.add(Dense(units2, activation='relu'))
    m.add(BatchNormalization())
    if dropout > 0:
        m.add(Dropout(dropout * 0.5))
    m.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return m

def calibrate_platt(raw_probs, y_true):
    """Return trained sklearn logistic calibrator mapping raw_probs -> calibrated probs"""
    lr = LogisticRegression(max_iter=2000)
    lr.fit(raw_probs.reshape(-1,1), y_true)
    return lr

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Analyseur Hippique IA Pro - DL", layout="wide")
st.title("🏇 Analyseur Hippique IA Pro — Deep Learning + Geny Scraper")

st.markdown("""
**Mode d'emploi rapide**
- Colle l'URL des partants Geny (ou uploade la page HTML),
- (Optionnel) colle l'URL stats pour historique ou uploade HTML,
- Prépare les features automatiquement, puis entraîne ou charge le modèle Keras,
- Calibre, prédit et exporte les pronostics.
""")

# Input: URLs or HTML uploads
col1, col2 = st.columns([2,1])
with col1:
    url_partants = st.text_input("URL - Partants (Geny) / Lien page partants", placeholder="https://www.geny.com/partants-pmu/...")
    uploaded_html_partants = st.file_uploader("Ou upload HTML Partants", type=['html','htm'], help="Si Geny bloque, télécharge la page (Enregistrer sous...) et upload ici.")
with col2:
    url_stats = st.text_input("URL - Stats/Historique (Geny)", placeholder="https://www.geny.com/stats-pmu?id_course=...")
    uploaded_html_stats = st.file_uploader("Ou upload HTML Stats", type=['html','htm'])

extract_btn = st.button("🔎 Extraire & Préparer features")

# Container placeholders
partants_df = None
stats_df = None
prep_df = None
feat_cols = []

if extract_btn:
    # PARTANTS: prefer uploaded file if provided
    if uploaded_html_partants is not None:
        raw = uploaded_html_partants.read().decode('utf-8', errors='ignore')
        save_debug_html("uploaded_partants", raw)
        soup = BeautifulSoup(raw, 'html.parser')
        # try to parse the first table present with expected headers
        tbl = parse_table_by_headers(soup, ['nom','cote','poids','musique','num'])
        if tbl:
            rows = []
            for tr in tbl.find_all('tr'):
                cols = [td.get_text(" ", strip=True) for td in tr.find_all(['td','th'])]
                if len(cols)>1:
                    rows.append(cols)
            if len(rows) > 1:
                header = [h.lower() for h in rows[0]]
                # build df
                parsed = []
                for r in rows[1:]:
                    # naive mapping by index
                    parsed.append(dict(zip(header, r + ['']*(len(header)-len(r)))))
                partants_df = pd.DataFrame(parsed)
                # try to rename keys to expected French keys
                rename_map = {}
                for c in partants_df.columns:
                    lc = c.lower()
                    if 'nom' in lc or 'name' in lc:
                        rename_map[c] = 'Nom'
                    if 'cote' in lc:
                        rename_map[c] = 'Cote'
                    if 'poids' in lc or 'weight' in lc:
                        rename_map[c] = 'Poids'
                    if 'musique' in lc or 'music' in lc:
                        rename_map[c] = 'Musique'
                    if 'num' in lc or 'corde' in lc:
                        rename_map[c] = 'Numéro de corde'
                partants_df = partants_df.rename(columns=rename_map)
            else:
                st.warning("HTML upload partants : format tableau non reconnu.")
        else:
            st.warning("HTML upload partants : aucun tableau identifié.")
    elif url_partants:
        partants_df, msg = scrape_geny_partants(url_partants)
        if partants_df is None:
            st.error(msg)
        else:
            st.success(msg)
    else:
        st.info("Fournir une URL partants ou uploader la page HTML.")
    # STATS
    if uploaded_html_stats is not None:
        raw = uploaded_html_stats.read().decode('utf-8', errors='ignore')
        save_debug_html("uploaded_stats", raw)
        soup = BeautifulSoup(raw, 'html.parser')
        tbl = parse_table_by_headers(soup, ['date','nom','position','cote'])
        if tbl:
            rows = []
            for tr in tbl.find_all('tr'):
                cols = [td.get_text(" ", strip=True) for td in tr.find_all(['td','th'])]
                if len(cols)>1:
                    rows.append(cols)
            if len(rows) > 1:
                header = [h.lower() for h in rows[0]]
                parsed = []
                for r in rows[1:]:
                    parsed.append(dict(zip(header, r + ['']*(len(header)-len(r)))))
                stats_df = pd.DataFrame(parsed)
                # rename
                rename_map = {}
                for c in stats_df.columns:
                    lc = c.lower()
                    if 'nom' in lc:
                        rename_map[c] = 'Nom'
                    if 'position' in lc or 'arriv' in lc or 'rang' in lc:
                        rename_map[c] = 'Position'
                    if 'cote' in lc:
                        rename_map[c] = 'Cote'
                stats_df = stats_df.rename(columns=rename_map)
            else:
                st.warning("HTML upload stats : tableau non reconnu.")
        else:
            st.warning("HTML upload stats : aucun tableau identifié.")
    elif url_stats:
        stats_df, msg = scrape_geny_stats(url_stats)
        if stats_df is None:
            st.info(msg)
        else:
            st.success(msg)

    # show extracted
    if partants_df is not None:
        st.subheader("Partants extraits")
        st.dataframe(partants_df.head(50), use_container_width=True)
    if stats_df is not None:
        st.subheader("Historique extrait (stats)")
        st.dataframe(stats_df.head(50), use_container_width=True)

    # Prepare features
    if partants_df is not None:
        prep_df, feat_cols = prepare_features_from_partants(partants_df, stats_df)
        st.subheader("Features préparées")
        st.write("Features utilisées:", feat_cols)
        st.dataframe(prep_df.head(50), use_container_width=True)
    else:
        st.stop()

# -----------------------------------
# Model training / load / predict UI
# -----------------------------------
st.markdown("---")
st.header("🔧 Entraînement / Chargement modèle (Deep Learning)")

colA, colB, colC = st.columns(3)
with colA:
    load_existing = st.button("Charger modèle existant (model/scaler/calib)")
with colB:
    train_local = st.button("Entraîner modèle local (sur historique fourni)")
with colC:
    quick_predict_btn = st.button("Prédire (utiliser modèle chargé/entrainé)")

model = None
scaler = None
calibrator = None

# Load existing
if load_existing:
    errors = []
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            st.success("Modèle Keras & scaler chargés.")
            if os.path.exists(CALIB_PATH):
                calibrator = joblib.load(CALIB_PATH)
                st.success("Calibrateur chargé.")
        except Exception as e:
            st.error(f"Erreur chargement: {e}")
    else:
        st.warning("Modèle/scaler non trouvé sur disque.")

# Train local if requested
if train_local:
    # require stats_df with Position to build training set
    if stats_df is None or 'Position' not in stats_df.columns:
        st.error("Impossible d'entraîner localement: aucun historique exploitable (stats avec 'Position' requis).")
    else:
        # Build training dataset from stats_df
        hist = stats_df.copy()
        # normalize
        hist['Nom'] = hist['Nom'].astype(str).str.strip()
        hist['Position_num'] = hist['Position'].apply(lambda x: safe_float(x, default=np.nan))
        hist_valid = hist[hist['Position_num'].notna()].copy()
        if len(hist_valid) < 30:
            st.warning("Historique trop petit (<30 lignes) — le modèle risque de sur-apprendre. Voulez-vous continuer ?")
        # Try to extract features for hist: odds & music if available
        # If stats_df includes 'Cote' or 'Musique', we can derive features; otherwise we train only on odds_inv
        # Basic approach: use odds if present
        if 'Cote' in hist_valid.columns:
            hist_valid['odds_numeric'] = hist_valid['Cote'].apply(lambda x: safe_float(x, default=np.nan)).fillna(hist_valid['Cote'].median() if hist_valid['Cote'].notna().any() else 5.0)
        else:
            hist_valid['odds_numeric'] = 5.0
        hist_valid['odds_inv'] = 1.0 / (hist_valid['odds_numeric'] + 0.1)
        # target: podium
        hist_valid['target'] = (hist_valid['Position_num'] <= 3).astype(int)
        # features for hist: if music exists, extract features from stats rows; else use odds_inv only
        X_hist = hist_valid[['odds_inv']].fillna(0.0).values
        y_hist = hist_valid['target'].values
        # Split
        Xtr, Xval, ytr, yval = train_test_split(X_hist, y_hist, test_size=0.2, random_state=42, stratify=y_hist if len(np.unique(y_hist))>1 else None)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xval_s = scaler.transform(Xval)
        # Build small keras
        input_dim = Xtr_s.shape[1]
        units1 = st.slider("Units couche 1", 16, 256, 64)
        units2 = st.slider("Units couche 2", 8, 128, 32)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        lr = float(st.text_input("LR (ex: 0.001)", "0.001"))
        m = build_keras_model(input_dim, units1=units1, units2=units2, dropout=dropout, lr=lr)
        st.write("Architecture :")
        m.summary(print_fn=lambda x: st.text(x))
        early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = m.fit(Xtr_s, ytr, validation_data=(Xval_s, yval), epochs=100, batch_size=16, callbacks=[early], verbose=1)
        # evaluation
        yval_pred = m.predict(Xval_s).ravel()
        auc_val = roc_auc_score(yval, yval_pred) if len(np.unique(yval))>1 else np.nan
        logloss_val = log_loss(yval, np.clip(yval_pred, 1e-6, 1-1e-6))
        st.write(f"Validation AUC: {auc_val:.4f} | LogLoss: {logloss_val:.4f}")
        # save
        m.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        st.success("Modèle & scaler sauvegardés.")
        # Platt calibration
        do_cal = st.checkbox("Appliquer calibration Platt sur set validation ?", value=True)
        if do_cal:
            calibrator = calibrate_platt(yval_pred, yval)
            joblib.dump(calibrator, CALIB_PATH)
            st.success("Calibrateur (Platt) sauvegardé.")

        # assign to model/scaler variables for immediate use
        model = m

# Quick predict using loaded/trained model or fallback deterministic score
if quick_predict_btn:
    # require prep_df
    if 'prep_df' not in locals() or prep_df is None:
        st.error("Aucune donnée préparée : extraire les partants d'abord.")
    else:
        df_ready = prep_df.copy()
        feats = feat_cols
        X_pred = df_ready[feats].values
        # if model/scaler exist, use them
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                model = load_model(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)
                if os.path.exists(CALIB_PATH):
                    calibrator = joblib.load(CALIB_PATH)
                st.success("Modèle & scaler chargés pour prédiction.")
            except Exception as e:
                st.error(f"Erreur chargement: {e}")
                model = None
        if model is not None and scaler is not None:
            Xs = scaler.transform(X_pred)
            raw = model.predict(Xs).ravel()
            if calibrator is not None:
                try:
                    probs = calibrator.predict_proba(raw.reshape(-1,1))[:,1]
                except Exception:
                    probs = raw
            else:
                probs = raw
            df_ready['prob_podium'] = probs
            df_ready['rank_pred'] = df_ready['prob_podium'].rank(ascending=False)
            st.subheader("🏆 Classement prédit (modèle DL)")
            st.dataframe(df_ready[['Nom','Cote','prob_podium','rank_pred']].sort_values('prob_podium', ascending=False), use_container_width=True)
            # If true labels (Resultat) available, show AUC
            if 'Resultat' in df_ready.columns:
                try:
                    df_ready['target'] = (df_ready['Resultat'] <= 3).astype(int)
                    if len(df_ready['target'].unique())>1:
                        auc = roc_auc_score(df_ready['target'], df_ready['prob_podium'])
                        st.metric("AUC (sur data chargée)", f"{auc:.3f}")
                except Exception:
                    pass
            # visualization
            fig = px.scatter(df_ready, x='Cote', y='prob_podium', color='prob_podium', text='Nom', title='Probabilité prédite vs Cote')
            st.plotly_chart(fig, use_container_width=True)
            # export
            csv = df_ready.to_csv(index=False)
            st.download_button("Télécharger prédictions (CSV)", csv, file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            st.warning("Aucun modèle DL disponible : on applique un score heuristique basé sur la cote.")
            # fallback: simple normalized score: odds_inv w/ small weights to recent_form/historical
            df_ready['heur_score'] = (df_ready['odds_inv']*0.5 + df_ready['recent_form']*0.2 + df_ready['historical_podium_rate']*0.3)
            # normalize
            if df_ready['heur_score'].max() != df_ready['heur_score'].min():
                df_ready['heur_score'] = (df_ready['heur_score'] - df_ready['heur_score'].min()) / (df_ready['heur_score'].max() - df_ready['heur_score'].min())
            df_ready['rank_pred'] = df_ready['heur_score'].rank(ascending=False)
            st.subheader("🏆 Classement heuristique (fallback)")
            st.dataframe(df_ready[['Nom','Cote','heur_score','rank_pred']].sort_values('heur_score', ascending=False), use_container_width=True)
            csv = df_ready.to_csv(index=False)
            st.download_button("Télécharger heuristique (CSV)", csv, file_name=f"heur_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

# ---------------------------
# Optional: generate Quinté combos weighted by probabilities
# ---------------------------
st.markdown("---")
st.header("🎲 Génération combinaisons (Quinté / e-trio) - Optionnel")

if 'prep_df' in locals() and prep_df is not None:
    df_q = prep_df.copy()
    # need probability column
    if 'prob_podium' not in df_q.columns:
        # if model predicted earlier, maybe stored; compute simple normalized score
        if 'score_pred' in df_q.columns:
            df_q['prob_podium'] = df_q['score_pred']
        else:
            df_q['prob_podium'] = (1.0 / (df_q['odds_numeric'] + 0.1))
            if df_q['prob_podium'].max() != df_q['prob_podium'].min():
                df_q['prob_podium'] = (df_q['prob_podium'] - df_q['prob_podium'].min()) / (df_q['prob_podium'].max() - df_q['prob_podium'].min())
    # top-k horses by prob
    topK = st.slider("Choisir top-K pour combinaisons (ex: 6 pour Quinté)", 5, min(10, max(5, len(df_q))), value=min(6, len(df_q)))
    # pick topK indices
    df_sorted = df_q.sort_values('prob_podium', ascending=False).reset_index(drop=True)
    top_horses = df_sorted.head(topK)
    st.markdown(f"Top-{topK} sélectionnés pour générer combinaisons ({len(top_horses)} chevaux).")
    # simple generation: list all ordered 5-combinations (may be large) — we limit by topK
    import itertools
    # generate quinté (ordered 5) combinations (permutations) — but for performance use combinations then permutations? simpler: use permutations of length 5
    comb_type = st.selectbox("Type de combinaison", ["Quinté (ordered 5)", "Trio (ordered 3)", "Top2 unordered"])
    if st.button("Générer combinaisons et espérance simple"):
        horses = top_horses['Nom'].tolist()
        probs_map = dict(zip(top_horses['Nom'], top_horses['prob_podium']))
        combos = []
        if comb_type == "Quinté (ordered 5)":
            # permutations length 5
            perms = list(itertools.permutations(horses, 5))
            # For each permutation compute naive joint probability = prod(prob_i normalized)
            # But that's naive (events dependent). Use product as heuristic ranking metric.
            for p in perms:
                prob_joint = np.prod([probs_map[h] for h in p])
                combos.append((p, prob_joint))
            combos_sorted = sorted(combos, key=lambda x: x[1], reverse=True)[:200]  # show top200
            st.write(f"{len(perms)} permutations générées (top 200 affichés).")
        elif comb_type == "Trio (ordered 3)":
            perms = list(itertools.permutations(horses, 3))
            for p in perms:
                prob_joint = np.prod([probs_map[h] for h in p])
                combos.append((p, prob_joint))
            combos_sorted = sorted(combos, key=lambda x: x[1], reverse=True)[:200]
            st.write(f"{len(perms)} permutations générées (top 200 affichés).")
        else:  # Top2 unordered
            combs = list(itertools.combinations(horses, 2))
            for p in combs:
                prob_joint = probs_map[p[0]] * probs_map[p[1]]
                combos.append((p, prob_joint))
            combos_sorted = sorted(combos, key=lambda x: x[1], reverse=True)[:200]
            st.write(f"{len(combs)} combinaisons générées (top 200 affichés).")
        out_df = pd.DataFrame([{'comb': ' > '.join(list(c)), 'score': s} for c,s in combos_sorted])
        st.dataframe(out_df.head(200), use_container_width=True)
        csv = out_df.to_csv(index=False)
        st.download_button("Télécharger combinaisons (CSV)", csv, file_name=f"combos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
else:
    st.info("Prépare d'abord les partants pour générer des combinaisons.")

# ---------------------------
# footer notes
# ---------------------------
st.markdown("---")
st.markdown("""
**Notes importantes**
- Pour un modèle DL fiable : **beaucoup** d'historiques (idéalement milliers de lignes) avec `Resultat`/`Position` sont nécessaires.
- Si Geny bloque le scraping, sauvegarde la page en HTML (clic droit → Enregistrer sous) et upload.
- Les probabilités du réseau doivent être **calibrées** (Platt) pour être fiables ; le calibrateur est stocké dans `hippo_calibrator.joblib`.
- Respecte les conditions d'utilisation du site Geny pour le scraping.
""")
