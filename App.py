# app_geny_dl_final.py
"""
🏇 Analyseur Hippique IA Pro — Final (Streamlit)
Base : script original de l'utilisateur (scraping gardé)
Ajouts : Deep Learning (Keras), calibration Platt, sauvegarde/chargement, anti-doublons, sécurité
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import time
from datetime import datetime
import joblib

# plotting
import plotly.express as px
import plotly.graph_objects as go

# sklearn
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, log_loss

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# Paths & constants
# ---------------------------
MODEL_PATH = "hippo_dl_model.h5"
SCALER_PATH = "hippo_scaler.joblib"
CALIB_PATH = "hippo_calibrator.joblib"
SCRAPE_DEBUG_DIR = "scrape_debug"
os.makedirs(SCRAPE_DEBUG_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ---------------------------
# --- Keep original scraping functions (from user's script)
# ---------------------------
@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        table = soup.find('table')
        if not table:
            # save debug html
            fn = os.path.join(SCRAPE_DEBUG_DIR, f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            with open(fn, 'wb') as f:
                f.write(response.content)
            return None, f"Aucun tableau trouvé (debug saved: {fn})"
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:
                horses_data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),
                    "Poids": cols[-2].get_text(strip=True) if len(cols) > 4 else "60.0",
                    "Musique": cols[2].get_text(strip=True) if len(cols) > 5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols) > 6 else "",
                })
        if not horses_data:
            return None, "Aucune donnée extraite"
        return pd.DataFrame(horses_data), "Succès"
    except Exception as e:
        return None, f"Erreur: {str(e)}"

# ---------------------------
# small helpers
# ---------------------------
def save_debug_html(tag, html_text):
    fn = os.path.join(SCRAPE_DEBUG_DIR, f"debug_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    with open(fn, 'w', encoding='utf-8') as f:
        f.write(html_text)
    return fn

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return float(cleaned)
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str) or str(poids_str).strip() == '':
        return 60.0
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(match.group(1).replace(',', '.')) if match else 60.0

def extract_music_features(music_str):
    """From original script: convert 'Musique' string to features"""
    if pd.isna(music_str) or music_str == '':
        return {
            'wins': 0, 'places': 0, 'total_races': 0,
            'win_rate': 0, 'place_rate': 0, 'consistency': 0,
            'recent_form': 0, 'best_position': 10,
            'avg_position': 8, 'position_variance': 5
        }
    music = str(music_str)
    positions = [int(c) for c in music if c.isdigit() and int(c) > 0]
    if not positions:
        return {
            'wins': 0, 'places': 0, 'total_races': 0,
            'win_rate': 0, 'place_rate': 0, 'consistency': 0,
            'recent_form': 0, 'best_position': 10,
            'avg_position': 8, 'position_variance': 5
        }
    total = len(positions)
    wins = positions.count(1)
    places = sum(1 for p in positions if p <= 3)
    recent = positions[:3]
    recent_form = sum(1/p for p in recent) / len(recent) if recent else 0
    consistency = 1 / (np.std(positions) + 1) if len(positions) > 1 else 0
    return {
        'wins': wins,
        'places': places,
        'total_races': total,
        'win_rate': wins / total if total > 0 else 0,
        'place_rate': places / total if total > 0 else 0,
        'consistency': consistency,
        'recent_form': recent_form,
        'best_position': min(positions),
        'avg_position': np.mean(positions),
        'position_variance': np.var(positions)
    }

# ---------------------------
# Feature preparation (kept close to original)
# ---------------------------
def prepare_advanced_features(df, race_type="PLAT"):
    # replicate original features creation but keep names stable
    features = pd.DataFrame(index=df.index)
    features['odds_numeric'] = df['odds_numeric']
    features['odds_inv'] = 1 / (df['odds_numeric'] + 0.1)
    features['log_odds'] = np.log1p(df['odds_numeric'])
    features['sqrt_odds'] = np.sqrt(df['odds_numeric'])
    features['odds_squared'] = df['odds_numeric'] ** 2

    features['draw'] = df['draw_numeric']
    max_draw = df['draw_numeric'].max() if df['draw_numeric'].max() > 0 else 1
    features['draw_normalized'] = df['draw_numeric'] / max_draw

    # optimal draws from original CONFIGS (simple local mapping)
    CONFIGS = {
        "PLAT": {"optimal_draws":[1,2,3,4], "weight_importance":0.25},
        "ATTELE_AUTOSTART": {"optimal_draws":[4,5,6], "weight_importance":0.05},
        "ATTELE_VOLTE": {"optimal_draws":[], "weight_importance":0.05}
    }
    optimal_draws = CONFIGS.get(race_type, CONFIGS['PLAT'])['optimal_draws']
    features['optimal_draw'] = df['draw_numeric'].apply(lambda x: 1 if x in optimal_draws else 0)
    if optimal_draws:
        features['draw_distance_optimal'] = df['draw_numeric'].apply(lambda x: min([abs(x - opt) for opt in optimal_draws]))
    else:
        features['draw_distance_optimal'] = 0

    features['weight_kg'] = df['weight_kg']
    features['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-6)
    features['weight_rank'] = df['weight_kg'].rank()
    weight_importance = CONFIGS.get(race_type, CONFIGS['PLAT'])['weight_importance']
    features['weight_advantage'] = (df['weight_kg'].max() - df['weight_kg']) * weight_importance

    features['age'] = df.get('age', 4.5)
    features['age_squared'] = features['age'] ** 2
    features['age_optimal'] = features['age'].apply(lambda x: 1 if 4 <= x <= 6 else 0)

    # music features
    if 'Musique' in df.columns:
        music_features = df['Musique'].apply(extract_music_features)
        # expand into DataFrame
        mf_df = pd.DataFrame(list(music_features))
        # rename keys to avoid collisions later
        mf_df.columns = [f"music_{c}" for c in mf_df.columns]
        features = pd.concat([features, mf_df.reset_index(drop=True)], axis=1)
    else:
        for key in ['wins', 'places', 'total_races', 'win_rate', 'place_rate', 'consistency', 'recent_form', 'best_position', 'avg_position', 'position_variance']:
            features[f'music_{key}'] = 0

    # interactions
    features['odds_draw_interaction'] = features['odds_inv'] * features['draw_normalized']
    features['odds_weight_interaction'] = features['log_odds'] * features['weight_normalized']
    features['age_weight_interaction'] = features['age'] * features['weight_kg']
    features['form_odds_interaction'] = features['music_recent_form'] * features['odds_inv']
    features['consistency_weight'] = features['music_consistency'] * features['weight_advantage']

    features['odds_rank'] = df['odds_numeric'].rank()
    features['odds_percentile'] = df['odds_numeric'].rank(pct=True)
    features['weight_percentile'] = df['weight_kg'].rank(pct=True)

    features['odds_z_score'] = (df['odds_numeric'] - df['odds_numeric'].mean()) / (df['odds_numeric'].std() + 1e-6)
    features['is_favorite'] = (df['odds_numeric'] == df['odds_numeric'].min()).astype(int)
    features['is_outsider'] = (df['odds_numeric'] > df['odds_numeric'].quantile(0.75)).astype(int)

    features['field_size'] = len(df)
    features['competitive_index'] = df['odds_numeric'].std() / (df['odds_numeric'].mean() + 1e-6)

    # fillna
    return features.fillna(0)

# ---------------------------
# Deep Learning model helpers
# ---------------------------
def build_keras_model(input_dim, units1=64, units2=32, dropout=0.2, lr=1e-3):
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(units1, activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(units2, activation='relu'))
    model.add(BatchNormalization())
    if dropout > 0:
        model.add(Dropout(dropout*0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])
    return model

def train_keras_and_calibrate(X, y, params=None):
    """
    Train Keras model with early stopping and return (model, scaler, calibrator, metrics)
    params: dict with hyperparams
    """
    if params is None:
        params = {}
    scaler = params.get('scaler', StandardScaler())
    Xs = scaler.fit_transform(X)

    # split train/val
    X_train, X_val, y_train, y_val = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None)

    input_dim = X_train.shape[1]
    model = build_keras_model(input_dim,
                              units1=params.get('units1', 64),
                              units2=params.get('units2', 32),
                              dropout=params.get('dropout', 0.2),
                              lr=params.get('lr', 1e-3))

    early = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=0)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=params.get('epochs', 200),
              batch_size=params.get('batch_size', 16), callbacks=[early], verbose=0)

    # predict on validation
    y_val_pred = model.predict(X_val).ravel()
    # calibrate with Platt scaling (logistic on probs)
    calibrator = None
    try:
        lr_clf = LogisticRegression(max_iter=2000)
        lr_clf.fit(y_val_pred.reshape(-1,1), y_val)
        calibrator = lr_clf
    except Exception:
        calibrator = None

    # compute metrics
    metrics = {}
    try:
        metrics['auc_val'] = roc_auc_score(y_val, calibrator.predict_proba(y_val_pred.reshape(-1,1))[:,1] if calibrator is not None else y_val_pred)
        metrics['logloss_val'] = log_loss(y_val, np.clip(calibrator.predict_proba(y_val_pred.reshape(-1,1))[:,1] if calibrator is not None else y_val_pred, 1e-6, 1-1e-6))
    except Exception:
        metrics['auc_val'] = np.nan
        metrics['logloss_val'] = np.nan

    return model, scaler, calibrator, metrics

# ---------------------------
# Streamlit UI (based on original layout)
# ---------------------------
st.set_page_config(page_title="🏇 Analyseur Hippique IA Pro - DL final", page_icon="🏇", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.4rem; color: #1e3a8a; text-align:center; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA Pro — Deep Learning (Final)</h1>', unsafe_allow_html=True)
st.markdown("Version : Deep Learning remplace la régression. Basé sur ton script initial (scraping conservé).")

# Sidebar config (kept simple)
with st.sidebar:
    st.header("⚙️ Configuration")
    ml_epochs = st.number_input("Epochs (DL)", min_value=10, max_value=2000, value=200, step=10)
    ml_batch = st.selectbox("Batch size", [8,16,32,64], index=1)
    ml_lr = st.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.6f")
    ml_units1 = st.slider("Units couche 1", 16, 256, 64, step=16)
    ml_units2 = st.slider("Units couche 2", 8, 128, 32, step=8)
    ml_dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
    auto_save_model = st.checkbox("Sauvegarder modèle après entraînement", value=True)

# Tabs: keep the same as original
tab1, tab2, tab3 = st.tabs(["🌐 URL Analysis", "📁 Upload CSV", "🧪 Test Data"])

df_final = None

with tab1:
    st.subheader("🔍 Analyse d'URL de Course (Geny)")
    col1, col2 = st.columns([3,1])
    with col1:
        url = st.text_input("🌐 URL de la course (page partants):", placeholder="https://www.geny.com/partants-pmu/...")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyser", type="primary")
    if analyze_button and url:
        with st.spinner("🔄 Extraction des données..."):
            df, message = scrape_race_data(url)
            if df is not None:
                st.success(f"✅ {len(df)} chevaux extraits avec succès")
                st.dataframe(df.head(), use_container_width=True)
                df_final = df
            else:
                st.error(f"❌ {message}")

with tab2:
    st.subheader("📤 Upload de fichier CSV (partants)")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    if uploaded_file:
        try:
            df_final = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(df_final)} chevaux chargés")
            st.dataframe(df_final.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erreur de lecture: {e}")

with tab3:
    st.subheader("🧪 Données de Test (exemples)")
    col1, col2, col3 = st.columns(3)
    if col1.button("🏃 Test Plat"):
        df_final = pd.DataFrame({
            'Nom': ['Thunder Bolt','Lightning Star','Storm King','Rain Dance','Wind Walker','Fire Dancer','Ocean Wave'],
            'Numéro de corde': ['1','2','3','4','5','6','7'],
            'Cote': ['3.2','4.8','7.5','6.2','9.1','12.5','15.0'],
            'Poids': ['56.5','57.0','58.5','59.0','57.5','60.0','61.5'],
            'Musique': ['1a2a3a1a','2a1a4a3a','3a3a1a2a','1a4a2a1a','4a2a5a3a','5a3a6a4a','6a5a7a8a'],
            'Âge/Sexe': ['4H','5M','3F','6H','4M','5H','4F']
        })
        st.success("✅ Données PLAT test chargées (7 chevaux)")
        st.dataframe(df_final, use_container_width=True)
    if col2.button("🚗 Test Attelé"):
        df_final = pd.DataFrame({
            'Nom': ['Rapide Éclair','Foudre Noire','Vent du Nord','Tempête Rouge','Orage Bleu','Cyclone Vert'],
            'Numéro de corde': ['1','2','3','4','5','6'],
            'Cote': ['4.2','8.5','15.0','3.8','6.8','10.2'],
            'Poids': ['68.0','68.0','68.0','68.0','68.0','68.0'],
            'Musique': ['2a1a4a1a','4a3a2a5a','6a5a8a7a','1a2a1a3a','3a4a5a2a','5a6a4a8a'],
            'Âge/Sexe': ['5H','6M','4F','7H','5M','6H']
        })
        st.success("✅ Données ATTELÉ test chargées (6 chevaux)")
        st.dataframe(df_final, use_container_width=True)
    if col3.button("⭐ Test Premium"):
        df_final = pd.DataFrame({
            'Nom': ['Ace Impact','Torquator Tasso','Adayar','Tarnawa','Chrono Genesis','Mishriff','Love'],
            'Numéro de corde': ['1','2','3','4','5','6','7'],
            'Cote': ['3.2','4.8','7.5','6.2','9.1','5.5','11.0'],
            'Poids': ['59.5','59.5','59.5','58.5','58.5','59.0','58.0'],
            'Musique': ['1a1a2a1a','1a3a1a2a','2a1a4a1a','1a2a1a3a','3a1a2a1a','1a1a1a2a','2a3a1a4a'],
            'Âge/Sexe': ['4H','5H','4H','5F','5F','5H','4F']
        })
        st.success("✅ Données PREMIUM test chargées (7 chevaux)")
        st.dataframe(df_final, use_container_width=True)

# ---------- Main analysis ----------
if df_final is not None and len(df_final) > 0:
    st.markdown("---")
    st.header("🎯 Analyse & Prédiction (Deep Learning)")

    # --- Prepare base columns like original script ---
    df_prepared = df_final.copy()
    # Numeric conversions (keep original safe_convert behavior)
    df_prepared['odds_numeric'] = df_prepared['Cote'].apply(lambda x: safe_float(x, 999))
    df_prepared['draw_numeric'] = df_prepared['Numéro de corde'].apply(lambda x: int(re.sub(r'\D','',str(x))) if re.sub(r'\D','',str(x)) else 1)
    df_prepared['weight_kg'] = df_prepared['Poids'].apply(extract_weight) if 'Poids' in df_prepared.columns else 60.0

    # music features
    if 'Musique' in df_prepared.columns:
        music_features = df_prepared['Musique'].apply(extract_music_features).apply(pd.Series)
        # rename columns to avoid duplicates with later feature df
        music_features = music_features.rename(columns=lambda c: f"music_{c}")
        df_prepared = pd.concat([df_prepared.reset_index(drop=True), music_features.reset_index(drop=True)], axis=1)
    else:
        # add default music columns
        df_prepared['music_wins'] = 0; df_prepared['music_places'] = 0
        df_prepared['music_total_races'] = 0; df_prepared['music_win_rate'] = 0.0
        df_prepared['music_place_rate'] = 0.0; df_prepared['music_consistency'] = 0.0
        df_prepared['music_recent_form'] = 0.0; df_prepared['music_best_position'] = 10

    # If user provided stats URL manually, attempt to fetch historical labels (optional)
    st.info("ℹ️ Si tu veux entraîner le modèle avec des vrais labels, fournis l'historique (stats) ou un CSV historique contenant 'Resultat' (rang).")

    # Attempt to detect race type via weight heuristics (kept from original)
    weight_std = df_prepared['weight_kg'].std() if 'weight_kg' in df_prepared else 0
    weight_mean = df_prepared['weight_kg'].mean() if 'weight_kg' in df_prepared else 60
    if weight_std > 2.5:
        detected_type = "PLAT"
    elif weight_mean > 65 and weight_std < 1.5:
        detected_type = "ATTELE_AUTOSTART"
    else:
        detected_type = "PLAT"
    st.info(f"🧾 Type détecté: {detected_type}")

    # Build feature matrix using prepare_advanced_features
    X_features = prepare_advanced_features(df_prepared, race_type=detected_type)

    # Avoid duplicate column names between original df_prepared and X_features when concatenating
    # Rename X_features columns if they exist in df_prepared
    Xf = X_features.copy()
    overlap = [c for c in Xf.columns if c in df_prepared.columns]
    if overlap:
        Xf = Xf.rename(columns={c: f"{c}_feat" for c in overlap})

    # final dataset for ML
    ML_df = pd.concat([df_prepared.reset_index(drop=True), Xf.reset_index(drop=True)], axis=1)

    # anti-duplicate safety: remove duplicated columns keeping first occurrence
    if ML_df.columns.duplicated().any():
        dup_cols = ML_df.columns[ML_df.columns.duplicated()].tolist()
        st.warning(f"⚠️ Colonnes dupliquées détectées et supprimées: {dup_cols}")
        ML_df = ML_df.loc[:, ~ML_df.columns.duplicated()]

    # Define feature list for DL: prefer engineered features (Xf columns)
    feature_cols = [c for c in Xf.columns if c in ML_df.columns]
    if len(feature_cols) == 0:
        st.error("❌ Aucun feature ML disponible — vérifie la préparation des données")
        st.stop()

    st.info(f"🔬 {len(feature_cols)} features utilisées pour le modèle DL")

    # Try to obtain labels (target) from stats/history or uploaded CSV
    # Priority:
    # 1) If user provided a 'Resultat' column in uploaded df_final (historic CSV), use it
    # 2) Else, ask user to provide a stats URL (handled in a separate input)
    target_available = False
    if 'Resultat' in df_prepared.columns:
        ML_df['target'] = (ML_df['Resultat'].apply(lambda x: safe_float(x, default=np.nan)) <= 3).astype(int)
        if ML_df['target'].notna().sum() > 0:
            target_available = True

    # Provide optional stats URL input to retrieve historical labels (if not present)
    st.markdown("#### (Optionnel) Si tu as une page 'stats' (historique) pour entraîner le modèle, colle l'URL ci-dessous")
    stats_url = st.text_input("URL - Stats/Historique (Geny) - optionnel", placeholder="https://www.geny.com/stats-pmu?id_course=...")
    stats_uploaded = st.file_uploader("Ou upload la page HTML stats (optionnel)", type=["html","htm"])
    stats_df = None
    if stats_url or stats_uploaded:
        if stats_uploaded:
            raw = stats_uploaded.read().decode('utf-8', errors='ignore')
            # naive parsing: try to find table (re-use simple parse)
            soup = BeautifulSoup(raw, 'html.parser')
            table = soup.find('table')
            if table:
                rows = []
                for tr in table.find_all('tr'):
                    cols = [td.get_text(" ", strip=True) for td in tr.find_all(['td','th'])]
                    if len(cols) > 1:
                        rows.append(cols)
                if len(rows) >= 2:
                    header = [h.lower() for h in rows[0]]
                    parsed = []
                    for r in rows[1:]:
                        rowd = dict(zip(header, r + [''] * (len(header)-len(r))))
                        parsed.append(rowd)
                    stats_df = pd.DataFrame(parsed)
            else:
                st.warning("Le HTML uploadé ne contient pas de tableau détectable.")
        else:
            # try scraping with the original scrape_race_data logic but for stats page we attempt a simple GET
            try:
                html, err = requests.get(stats_url, headers=HEADERS, timeout=10), None
            except Exception as e:
                html, err = None, str(e)
            if html is not None and hasattr(html, 'text'):
                raw = html.text
                # quick heuristic: find lines with dates/positions
                soup = BeautifulSoup(raw, 'html.parser')
                text_lines = soup.get_text("\n", strip=True).split("\n")
                candidates = [l for l in text_lines if re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b', l)]
                parsed = []
                for l in candidates:
                    # extract name and position heuristically
                    pos_m = re.search(r'\b([1-9]|1[0-9])\b', l)
                    date_m = re.search(r'(\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', l)
                    parsed.append({'raw': l, 'Date': date_m.group(1) if date_m else '', 'Position': pos_m.group(1) if pos_m else ''})
                if parsed:
                    stats_df = pd.DataFrame(parsed)

        # if stats_df constructed contains 'Position' & 'Nom' try to map to ML_df
        if stats_df is not None and 'Position' in stats_df.columns and 'Nom' in stats_df.columns:
            stats_df['Nom_clean'] = stats_df['Nom'].astype(str).str.strip().str.upper()
            stats_df['Position_num'] = stats_df['Position'].apply(lambda x: safe_float(x, default=np.nan))
            # aggregate podium rate per name
            s_valid = stats_df[stats_df['Position_num'].notna()]
            if len(s_valid) > 0:
                podium = s_valid.groupby('Nom_clean').apply(lambda g: (g['Position_num'] <= 3).mean()).rename('hist_podium_rate').reset_index()
                podium_map = dict(zip(podium['Nom_clean'], podium['hist_podium_rate']))
                ML_df['historical_podium_rate'] = ML_df['Nom'].astype(str).str.strip().str.upper().map(podium_map).fillna(0.0)
                # optional: if stats_df contains Resultat for current race (rare), we could label; skip complexity
                # but if stats_df has entries corresponding to current race with positions, map them
                mapped_positions = {}
                for idx,row in ML_df.iterrows():
                    name = str(row.get('Nom','')).strip().upper()
                    # try to find a direct match in stats_df for the same date? complicated -> skip
                # we set target_available flag only if user provided Resultat in uploaded CSV earlier
        else:
            st.info("ℹ️ Aucun historique exploitable automatiquement trouvé dans la page stats fournie (format heuristique). Si tu veux entraîner avec labels, fournis un CSV historique contenant 'Resultat' (rang).")

    # Training: only if user asks
    st.markdown("#### Entraînement du modèle Deep Learning")
    train_button = st.button("▶️ Entraîner modèle DL maintenant (si labels disponibles)")

    model = None
    scaler = None
    calibrator = None
    trained_metrics = {}

    if train_button:
        # If ML_df contains 'target' (from uploaded historic CSV), we can train; else fallback to heuristic training (not recommended)
        if 'target' in ML_df and ML_df['target'].nunique() > 1:
            X = ML_df[feature_cols].values
            y = ML_df['target'].values
            params = {
                'scaler': StandardScaler(),
                'units1': ml_units1, 'units2': ml_units2,
                'dropout': ml_dropout, 'lr': ml_lr,
                'epochs': int(ml_epochs), 'batch_size': int(ml_batch)
            }
            with st.spinner("🤖 Entraînement du modèle DL..."):
                model, scaler, calibrator, metrics = train_keras_and_calibrate(X, y, params=params)
                trained_metrics = metrics
                st.success("✅ Entraînement terminé")
                st.write("🔎 Metrics validation:", metrics)
                # save if requested
                if auto_save_model:
                    try:
                        model.save(MODEL_PATH)
                        joblib.dump(scaler, SCALER_PATH)
                        if calibrator is not None:
                            joblib.dump(calibrator, CALIB_PATH)
                        st.success("💾 Modèle, scaler et calibrateur sauvegardés.")
                    except Exception as e:
                        st.warning(f"Erreur sauvegarde: {e}")
        else:
            # Fallback: no real labels. We can warn user and create a pseudo-target to allow training (not ideal)
            st.warning("Aucun label réel (target) détecté. Le modèle sera entraîné sur une cible synthétique dérivée des cotes (FALLBACK).")
            # Create weak pseudo-target: favorite = odds<median -> 1 else 0 (not recommended for real use)
            pseudo = (ML_df['odds_numeric'] <= ML_df['odds_numeric'].median()).astype(int)
            X = ML_df[feature_cols].values
            y = pseudo.values
            params = {'scaler': StandardScaler(), 'units1': ml_units1, 'units2': ml_units2, 'dropout': ml_dropout, 'lr':ml_lr, 'epochs': int(ml_epochs), 'batch_size': int(ml_batch)}
            with st.spinner("🛠️ Entraînement (fallback synthétique) ..."):
                model, scaler, calibrator, metrics = train_keras_and_calibrate(X, y, params=params)
                trained_metrics = metrics
                st.success("✅ Entraînement (fallback) terminé — attention aux limitations d'utilisation")
                if auto_save_model:
                    try:
                        model.save(MODEL_PATH)
                        joblib.dump(scaler, SCALER_PATH)
                        if calibrator is not None:
                            joblib.dump(calibrator, CALIB_PATH)
                        st.success("💾 Modèle & scaler sauvegardés.")
                    except Exception as e:
                        st.warning(f"Erreur sauvegarde: {e}")

    # Option to load existing model
    if st.button("🔁 Charger modèle existant (si présent sur disque)"):
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                model = load_model(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)
                if os.path.exists(CALIB_PATH):
                    calibrator = joblib.load(CALIB_PATH)
                st.success("✅ Modèle, scaler et calibrateur chargés")
            else:
                st.warning("Aucun modèle/scaler trouvé sur disque.")
        except Exception as e:
            st.error(f"Erreur chargement modèle: {e}")

    # Prediction: use model if available
    st.markdown("#### Prédire & Classer les partants")
    predict_button = st.button("🎯 Prédire maintenant (si modèle disponible)")

    if predict_button:
        if model is None or scaler is None:
            # try to auto-load if present
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                try:
                    model = load_model(MODEL_PATH)
                    scaler = joblib.load(SCALER_PATH)
                    if os.path.exists(CALIB_PATH):
                        calibrator = joblib.load(CALIB_PATH)
                    st.success("Chargé modèle depuis disque.")
                except Exception as e:
                    st.error(f"Erreur chargement automatique: {e}")
                    model = None
            else:
                st.warning("Aucun modèle disponible. Entraîne d'abord le modèle ou fournis un modèle sauvegardé.")
        if model is not None and scaler is not None:
            X_pred = ML_df[feature_cols].values
            Xs = scaler.transform(X_pred)
            raw_preds = model.predict(Xs).ravel()
            if calibrator is not None:
                try:
                    preds = calibrator.predict_proba(raw_preds.reshape(-1,1))[:,1]
                except Exception:
                    preds = raw_preds
            else:
                preds = raw_preds
            ML_df['prob_podium'] = preds
            ML_df['score_final'] = ML_df['prob_podium']  # main ranking metric
            ML_df['rang'] = ML_df['score_final'].rank(ascending=False).astype(int)
            # Normalize for display
            if ML_df['score_final'].max() != ML_df['score_final'].min():
                ML_df['score_final_norm'] = (ML_df['score_final'] - ML_df['score_final'].min()) / (ML_df['score_final'].max() - ML_df['score_final'].min())
            else:
                ML_df['score_final_norm'] = ML_df['score_final']

            # anti-duplicate protection before display (just in case)
            if ML_df.columns.duplicated().any():
                dup = ML_df.columns[ML_df.columns.duplicated()].tolist()
                st.warning(f"Doublons colonnes détectés et supprimés: {dup}")
                ML_df = ML_df.loc[:, ~ML_df.columns.duplicated()]

            st.subheader("🏆 Classement Final (DL)")
            display_cols = ['rang', 'Nom', 'Cote', 'Poids'] if 'Poids' in ML_df.columns else ['rang','Nom','Cote']
            display_cols += ['score_final_norm', 'prob_podium'] if 'prob_podium' in ML_df.columns else []
            display_df = ML_df[display_cols].copy()
            display_df = display_df.sort_values('rang')
            # Format numeric
            if 'score_final_norm' in display_df.columns:
                display_df['Score'] = display_df['score_final_norm'].apply(lambda x: f"{x:.3f}")
                display_df = display_df.drop('score_final_norm', axis=1)
            if 'prob_podium' in display_df.columns:
                display_df['Confiance'] = display_df['prob_podium'].apply(lambda x: f"{x:.1%}")
                display_df = display_df.drop('prob_podium', axis=1)
            st.dataframe(display_df, use_container_width=True, height=420)

            # Top 5 visual boxes
            st.subheader("🥇 Top 5 Prédictions")
            for i in range(min(5, len(ML_df))):
                horse = ML_df.sort_values('rang').iloc[i]
                conf = horse.get('score_final', 0)
                emoji = "🟢" if conf >= 0.7 else ("🟡" if conf >= 0.4 else "🔴")
                st.markdown(f"""
                <div style="border-left:4px solid #f59e0b;padding:10px;background:#fff7ed;border-radius:8px;margin-bottom:8px">
                    <strong>{i+1}. {horse['Nom']}</strong><br>
                    📊 Cote: <strong>{horse.get('Cote','')}</strong> | 🎯 Score: <strong>{horse['score_final']:.3f}</strong><br>
                    {emoji} Confiance: <strong style="color:#1f2937">{conf:.1%}</strong>
                </div>
                """, unsafe_allow_html=True)

            # Plot
            st.subheader("📊 Visualisation : Probabilité prédite vs Cote")
            fig = px.scatter(ML_df, x='odds_numeric', y='score_final', text='Nom', color='score_final',
                             color_continuous_scale='Viridis', labels={'odds_numeric':'Cote', 'score_final':'Prob(podium)'})
            st.plotly_chart(fig, use_container_width=True)

            # Exports
            csv_data = ML_df.to_csv(index=False)
            st.download_button("📄 Télécharger CSV complet", csv_data,
                               f"pronostic_dl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        else:
            st.error("Aucun modèle disponible pour prédiction.")

    # Option: quick heuristic score if user doesn't want DL or DL not trained
    if st.button("⚖️ Générer un score heuristique (cotes + musique)"):
        sturdy_score = 1.0/(ML_df['odds_numeric']+0.1) * 0.6 + ML_df['music_recent_form'] * 0.25 + ML_df.get('historical_podium_rate', 0.0) * 0.15
        ML_df['heur_score'] = sturdy_score
        ML_df['heur_norm'] = (ML_df['heur_score'] - ML_df['heur_score'].min()) / (ML_df['heur_score'].max() - ML_df['heur_score'].min())
        df_sorted = ML_df.sort_values('heur_norm', ascending=False)
        st.dataframe(df_sorted[['Nom','Cote','heur_norm']].head(20), use_container_width=True)

# End of app
st.markdown("---")
st.caption("Notes : Pour obtenir un modèle réellement prédictif, entraine sur beaucoup d'historiques réels (Resultat/rang). Le fallback synthétique est uniquement pour tests locaux.")
