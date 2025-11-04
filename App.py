# app.py
"""
🏇 Analyseur Hippique IA Pro - Version améliorée complète Streamlit
- Gardez votre scraping Geny.fr existant (fonction scrape_race_data inchangée en interface)
- Ajouts :
    * Réseau de neurones Keras (léger, CPU-friendly)
    * Ensembliste + blending
    * Pondération dynamique des features via corrélations (Pearson/Spearman)
    * Scoring explicatif simplifié (contributions approximatives)
    * Sorties: classement, probabilités/podiums, combinaisons, export CSV/JSON
Notes:
- Pour Pydroid: installez tensorflow CPU. Example: pip install tensorflow==2.12.0 (selon compatibilité)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# --- Scikit-learn ML ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# --- TensorFlow / Keras (lightweight NN) ---
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# --- UI config ---
st.set_page_config(page_title="🏇 Analyseur Hippique IA Pro — v2", page_icon="🏇", layout="wide")

# --- Helper utilities ---
def safe_convert(value, convert_func, default=0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str):
        return 60.0
    match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(match.group(1).replace(',', '.')) if match else 60.0

# --- Keep your scraping: kept-compatible with your original function interface ---
@st.cache_data(ttl=300)
def scrape_race_data(url):
    """
    La fonction conserve ton scraping original (robustesse améliorée).
    Rend DataFrame ou (None, message).
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"
        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        table = soup.find('table')
        if not table:
            # Tentative: trouver tableau alternatif
            table = soup.find('div', {'class': 'table-responsive'})
            if table:
                table = table.find('table')
        if not table:
            return None, "Aucun tableau trouvé (structure HTML inattendue)"
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all(['td', 'th'])
            # heuristique : minimum 4 colonnes
            textcols = [c.get_text(strip=True) for c in cols if c.get_text(strip=True) != '']
            if len(textcols) >= 4:
                # tries to keep column mapping flexible
                num = textcols[0]
                name = textcols[1]
                # attempt to find odds and weight heuristically (last two columns)
                cote = textcols[-1]
                poids = textcols[-2] if len(textcols) > 4 else "60.0"
                # musique often in col 2 or 3
                musique = textcols[2] if len(textcols) > 3 else ""
                agesexe = textcols[3] if len(textcols) > 4 else ""
                horses_data.append({
                    "Numéro de corde": num,
                    "Nom": name,
                    "Cote": cote,
                    "Poids": poids,
                    "Musique": musique,
                    "Âge/Sexe": agesexe
                })
        if not horses_data:
            return None, "Aucune donnée extraite"
        return pd.DataFrame(horses_data), "Succès"
    except Exception as e:
        return None, f"Erreur scraping: {str(e)}"

# --- Feature extraction utilities ---
def extract_music_features(music_str):
    """
    Extraction simple mais robuste de la 'musique' (performances récentes).
    On renvoie counts, win_rate, place_rate, recent_form (3 courses).
    """
    if pd.isna(music_str) or music_str == '':
        return {'wins':0, 'places':0, 'total_races':0, 'win_rate':0.0, 'place_rate':0.0, 'recent_form':0.0, 'avg_pos':10.0}
    # keep only digits as positions
    s = str(music_str)
    positions = [int(ch) for ch in s if ch.isdigit() and int(ch) > 0]
    if not positions:
        return {'wins':0, 'places':0, 'total_races':0, 'win_rate':0.0, 'place_rate':0.0, 'recent_form':0.0, 'avg_pos':10.0}
    total = len(positions)
    wins = positions.count(1)
    places = sum(1 for p in positions if p <= 3)
    win_rate = wins / total
    place_rate = places / total
    recent = positions[:3]
    recent_form = sum([1.0 / p for p in recent]) / len(recent) if recent else 0.0
    avg_pos = np.mean(positions)
    return {'wins':wins, 'places':places, 'total_races':total, 'win_rate':win_rate, 'place_rate':place_rate, 'recent_form':recent_form, 'avg_pos':avg_pos}

def prepare_data(df):
    """
    Nettoyage initial: conversion cotes, poids, draw, musiques -> features brutes.
    """
    df = df.copy()
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    # musique
    music_feats = df['Musique'].fillna('').apply(extract_music_features).tolist()
    music_df = pd.DataFrame(music_feats)
    df = pd.concat([df.reset_index(drop=True), music_df.reset_index(drop=True)], axis=1)
    # age/sex
    if 'Âge/Sexe' in df.columns:
        df['age'] = df['Âge/Sexe'].astype(str).str.extract(r'(\d+)').astype(float).fillna(4.5)
        df['is_female'] = df['Âge/Sexe'].astype(str).str.contains('F', case=False, na=False).astype(int)
    else:
        df['age'] = 4.5
        df['is_female'] = 0
    # simple derived features:
    df['odds_inv'] = 1.0 / (df['odds_numeric'] + 0.1)
    df['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / (df['weight_kg'].std() + 1e-6)
    df['is_favorite'] = (df['odds_numeric'] == df['odds_numeric'].min()).astype(int)
    df['field_size'] = len(df)
    # safety: remove infinite / extreme odds
    df.loc[df['odds_numeric'] > 1000, 'odds_numeric'] = 1000
    df = df.reset_index(drop=True)
    return df

# --- Advanced features builder (used by ML models) ---
def build_features(df, race_type="PLAT"):
    """
    Construit un DataFrame X de features numériques prêtes pour ML.
    Applique aussi pondération dynamique des features via corrélations.
    """
    features = pd.DataFrame()
    features['odds_numeric'] = df['odds_numeric']
    features['odds_inv'] = df['odds_inv']
    features['draw'] = df['draw_numeric']
    features['weight_kg'] = df['weight_kg']
    features['age'] = df['age']
    features['is_female'] = df['is_female']
    # musique features
    for col in ['wins','places','total_races','win_rate','place_rate','recent_form','avg_pos']:
        if col in df.columns:
            features[f'music_{col}'] = df[col]
        else:
            features[f'music_{col}'] = 0
    # interactions simple
    features['odds_x_recent'] = features['odds_inv'] * features['music_recent_form']
    features['weight_x_age'] = features['weight_kg'] * features['age']
    features['odds_x_draw'] = features['odds_numeric'] * features['draw']
    # relative ranks
    features['odds_rank'] = features['odds_numeric'].rank()
    features['weight_rank'] = features['weight_kg'].rank()
    features.fillna(0, inplace=True)
    return features

# --- Feature correlation & dynamic weighting ---
def compute_feature_weights(X, y=None):
    """
    Si y fourni: calcule corrélations (pearson + spearman) pour estimer l'importance.
    Retourne dict feature -> weight normalisé [0,1].
    Si y non fourni: renvoie poids uniformes.
    """
    if y is None or len(y) != len(X):
        return {c:1.0 for c in X.columns}
    corr_scores = {}
    for c in X.columns:
        try:
            px = X[c].fillna(0).values
            # Pearson
            p = np.corrcoef(px, y)[0,1] if len(px)>1 else 0.0
            # Spearman (rank)
            try:
                from scipy.stats import spearmanr
                s = spearmanr(px, y).correlation
            except Exception:
                s = 0.0
            score = np.nan_to_num(0.5 * (abs(p) + abs(s)))
        except Exception:
            score = 0.0
        corr_scores[c] = float(max(score, 0.0))
    # normalize into [0.2, 1.0] to avoid zeroing features (floor improves stability)
    arr = np.array(list(corr_scores.values()))
    if arr.max() == arr.min():
        norm = {k:1.0 for k in corr_scores.keys()}
    else:
        mn = arr.min(); mx = arr.max()
        norm = {k: 0.2 + 0.8 * ((v - mn) / (mx - mn)) for k,v in corr_scores.items()}
    return norm

# --- Keras model factory (small) ---
def create_keras_model(input_dim, lr=1e-3, dropout=0.2):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(int(max(32, input_dim*2)), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.Dense(int(max(16, input_dim//1)), activation='relu'),
        layers.Dropout(dropout/2),
        layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
    return model

# --- Training & prediction pipeline ---
class HorseRacingModel:
    def __init__(self, random_state=42):
        self.rf = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=random_state, n_jobs=-1)
        self.gb = GradientBoostingRegressor(n_estimators=120, learning_rate=0.05, max_depth=4, random_state=random_state)
        self.ridge = Ridge(alpha=1.0)
        self.nn_model = None
        self.scaler = RobustScaler()
        self.feature_weights = {}
        self.trained = False
        self.cv_scores = {}

    def fit(self, X, y, use_nn=True, epochs=60, batch_size=16):
        """
        Entraîne RF, GB, Ridge, puis un NN léger (optionnel).
        y peut être target synthétique (score) ou probabilités réelles si labels disponibles.
        """
        # Save columns
        self.features = X.columns.tolist()
        # scale
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # feature dynamic weights (computed from X,y)
        self.feature_weights = compute_feature_weights(X_scaled, y)
        # apply feature weights by scaling columns (emphase)
        X_weighted = X_scaled.copy()
        for c,w in self.feature_weights.items():
            X_weighted[c] = X_weighted[c] * w

        # Fit base models
        self.rf.fit(X_weighted, y)
        self.gb.fit(X_weighted, y)
        self.ridge.fit(X_weighted, y)

        # Cross-val quick metrics (R2)
        kf = KFold(n_splits=min(5, max(2, len(X)//2)), shuffle=True, random_state=42)
        try:
            self.cv_scores['rf'] = np.mean(cross_val_score(self.rf, X_weighted, y, cv=kf, scoring='r2', n_jobs=-1))
            self.cv_scores['gb'] = np.mean(cross_val_score(self.gb, X_weighted, y, cv=kf, scoring='r2', n_jobs=-1))
        except Exception:
            self.cv_scores['rf'] = None
            self.cv_scores['gb'] = None

        # NN training (wrapped)
        if use_nn:
            tf.keras.backend.clear_session()
            self.nn_model = create_keras_model(input_dim=X_weighted.shape[1], lr=1e-3, dropout=0.25)
            early = keras.callbacks.EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
            self.nn_model.fit(X_weighted.values, y, epochs=epochs, batch_size=min(batch_size, max(4, len(X)//2)), verbose=0, callbacks=[early])
        else:
            self.nn_model = None

        # blending weights: based on CV performance if available
        # simple heuristic: favor ensemble (gb/rf) + nn if exists
        self.weights = {'rf':0.25, 'gb':0.35, 'ridge':0.05, 'nn':0.35 if self.nn_model else 0.0}
        self.trained = True
        return self

    def predict(self, X):
        """
        Retourne prediction score (continuum). Si trained=False -> zeros.
        """
        if not self.trained:
            return np.zeros(len(X))
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        # apply feature weights
        Xw = X_scaled.copy()
        for c,w in self.feature_weights.items():
            if c in Xw.columns:
                Xw[c] = Xw[c] * w
        preds = {}
        preds['rf'] = self.rf.predict(Xw)
        preds['gb'] = self.gb.predict(Xw)
        preds['ridge'] = self.ridge.predict(Xw)
        if self.nn_model:
            preds['nn'] = self.nn_model.predict(Xw.values).reshape(-1)
        else:
            preds['nn'] = np.zeros(len(Xw))

        # blending (normalized)
        final = (self.weights['rf']*preds['rf'] + self.weights['gb']*preds['gb'] + self.weights['ridge']*preds['ridge'] + self.weights['nn']*preds['nn'])
        # normalize to 0..1
        if final.max() != final.min():
            final_norm = (final - final.min()) / (final.max() - final.min())
        else:
            final_norm = np.zeros_like(final)
        return final_norm, preds

    def explain_contributions(self, X, top_k=6):
        """
        Approximative: permutation importance & per-row linearized contributions
        - Compute permutation importance globally (RF)
        - For per-row approx: standardized feature * global importance weight
        """
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        Xw = X_scaled.copy()
        for c,w in self.feature_weights.items():
            if c in Xw.columns:
                Xw[c] *= w

        # permutation importance (uses rf for speed)
        try:
            r = permutation_importance(self.rf, Xw, np.zeros(len(Xw)), n_repeats=10, random_state=42, n_jobs=-1)
            imp = dict(zip(Xw.columns, r.importances_mean))
            # normalize
            total = sum(abs(v) for v in imp.values()) + 1e-9
            imp_norm = {k: abs(v)/total for k,v in imp.items()}
        except Exception:
            imp_norm = {c: 1.0/len(Xw.columns) for c in Xw.columns}

        # per-row contributions (approx): feature_value * importance
        contribs = pd.DataFrame(0.0, index=X.index, columns=X.columns)
        for c,w in imp_norm.items():
            contribs[c] = Xw[c] * w
        # for readability, pick top_k features per row
        top_features_rows = []
        for idx in X.index:
            row = contribs.loc[idx]
            top = row.abs().sort_values(ascending=False).head(top_k).index.tolist()
            top_features_rows.append({f: float(row[f]) for f in top})
        return imp_norm, top_features_rows

# --- Utility: build target if no real labels ---
def build_target(df):
    """
    Si l'utilisateur fournit une colonne 'Result' ou 'Position' (1 = gagnant),
    on utilise un target binaire/probabiliste. Sinon on construit un target proxy:
    target = normalized (odds_inv * 0.5 + music_win_rate * 0.5) + bruit
    Ou mieux: inverse rank proxy if position available.
    """
    if 'Position' in df.columns:
        # smaller is better: 1->best
        pos = df['Position'].astype(float)
        # convert to score (higher = better): inverse rank normalized
        s = 1.0 / (pos + 1e-6)
        return (s - s.min()) / (s.max() - s.min())
    if 'Result' in df.columns:
        # 'Result' might be '1' for winner, else
        res = df['Result'].astype(float)
        return res
    # otherwise synthetic
    proxy = df['odds_inv'] * 0.6 + df.get('win_rate', df.get('music_win_rate', df.get('win_rate', 0))) * 0.4 if 'win_rate' in df.columns or 'music_win_rate' in df.columns else df['odds_inv']
    # normalize
    arr = np.array(proxy).astype(float)
    if arr.max() != arr.min():
        s = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        s = np.zeros_like(arr)
    # add tiny noise for model stability
    s = s + np.random.normal(0, 0.02, len(s))
    s = np.clip(s, 0, 1)
    return s

# --- Combinaisons simple generator (top-N combos) ---
def generate_combinations(df_ranked, kind='trio', topk=5):
    """
    Génère combos simples:
    - 'trio' : toutes permutations top3 from topk
    - 'quinte' : combinaisons top5 choose 5 if available -> returns 1 combination (top5)
    Return: list of lists of horse names / numbers
    """
    res = []
    if kind == 'trio':
        top = df_ranked.head(topk)
        names = top['Nom'].tolist()
        # return combinations of top3 among first topk (simple: choose combos without order)
        from itertools import combinations
        for comb in combinations(names, 3):
            res.append(list(comb))
    elif kind == 'quinte':
        top = df_ranked.head(max(5, topk))
        if len(top) >= 5:
            res.append(top['Nom'].head(5).tolist())
    return res

# --- Visualization helpers ---
def create_summary_fig(df_ranked):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Scores vs Cotes','Confiance / Rang'))
    fig.add_trace(go.Bar(x=df_ranked['Nom'], y=df_ranked['score_final'], name='Score'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ranked['Nom'], y=df_ranked['odds_numeric'], name='Cote', yaxis='y2', mode='markers'), row=1, col=1)
    fig.add_trace(go.Bar(x=df_ranked['Nom'], y=df_ranked.get('confidence', np.zeros(len(df_ranked))), name='Confiance'), row=1, col=2)
    fig.update_layout(height=450, showlegend=True)
    return fig

# === Streamlit UI & Main ===
def main():
    st.markdown("<h1 style='text-align:center;color:#1e3a8a'>🏇 Analyseur Hippique IA Pro — v2</h1>", unsafe_allow_html=True)
    st.markdown("Version améliorée : réseau neuronal léger + ensembliste + pondération dynamique des features")
    with st.sidebar:
        st.header("⚙️ Configuration")
        race_type = st.selectbox("Type de course", ["AUTO","PLAT","ATTELE_AUTOSTART","ATTELE_VOLTE"])
        use_nn = st.checkbox("Inclure NN (Keras)", value=True)
        nn_epochs = st.slider("Epochs NN", 10, 200, 60, 10)
        ml_blend_weight = st.slider("Poids ML vs Cotes (0=cotes,1=ML)", 0.0, 1.0, 0.7, 0.05)
        show_explain = st.checkbox("Afficher explication contributions (approx.)", value=True)
    tab1, tab2, tab3 = st.tabs(["🔗 URL", "📁 CSV upload", "🧪 Test Data"])

    df = None
    with tab1:
        st.subheader("Analyse d'une URL (Scraping)")
        url = st.text_input("URL de la page course (Geny / autre)", placeholder="https://...")
        if st.button("Analyser l'URL"):
            if not url:
                st.error("Veuillez renseigner une URL.")
            else:
                with st.spinner("Extraction ..."):
                    df_res, msg = scrape_race_data(url)
                    if df_res is None:
                        st.error(msg)
                    else:
                        st.success(f"{len(df_res)} chevaux extraits")
                        df = df_res.copy()
                        st.dataframe(df.head())

    with tab2:
        st.subheader("Upload CSV")
        uploaded = st.file_uploader("Fichier CSV (Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe, optionnel: Position/Result)", type=['csv'])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"{len(df)} chevaux chargés depuis CSV")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Erreur lecture CSV: {e}")

    with tab3:
        st.subheader("Données test rapides")
        if st.button("Charger jeu example Plat"):
            df = pd.DataFrame({
                'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker', 'Fire Dancer', 'Ocean Wave'],
                'Numéro de corde': ['1','2','3','4','5','6','7'],
                'Cote': ['3.2','4.8','7.5','6.2','9.1','12.5','15.0'],
                'Poids': ['56.5','57.0','58.5','59.0','57.5','60.0','61.5'],
                'Musique': ['1a2a3a1a','2a1a4a3a','3a3a1a2a','1a4a2a1a','4a2a5a3a','5a3a6a4a','6a5a7a8a'],
                'Âge/Sexe': ['4H','5M','3F','6H','4M','5H','4F']
            })
            st.success("Données test chargées")
            st.dataframe(df)

    if df is None:
        st.info("Chargez une course via URL ou CSV ou test data pour lancer l'analyse.")
        return

    # Prepare
    df_prep = prepare_data(df)
    st.markdown("### 🔎 Aperçu des données préparées")
    st.dataframe(df_prep[['Nom','Numéro de corde','Cote','odds_numeric','weight_kg','wins','win_rate']].fillna('').head(20))

    # Detect race type (simple heur)
    if race_type == 'AUTO':
        weight_std = df_prep['weight_kg'].std()
        if weight_std > 2.5:
            detected = "PLAT"
        else:
            detected = "ATTELE_AUTOSTART"
    else:
        detected = race_type
    st.info(f"Type détecté: {detected}")

    # Build X
    X = build_features(df_prep, detected)
    y = build_target(df_prep)  # target proxy or from true labels if present

    st.markdown("### 🧠 Entraînement ML")
    st.write(f"Nombre de features: {len(X.columns)}")
    # Wrap ML
    hrm = HorseRacingModel()
    with st.spinner("Entraînement modèles (RF/GB/Ridge/NN)..."):
        hrm.fit(X, y, use_nn=use_nn, epochs=nn_epochs, batch_size=8)

    # Predict
    scores_norm, preds_components = hrm.predict(X)
    # combine with odds (traditional) using ml_blend_weight
    traditional = (1.0 / (df_prep['odds_numeric'] + 0.1)).values
    # normalize traditional
    if traditional.max() != traditional.min():
        trad_norm = (traditional - traditional.min()) / (traditional.max() - traditional.min())
    else:
        trad_norm = np.zeros_like(traditional)
    final_score = ml_blend_weight * scores_norm + (1-ml_blend_weight) * trad_norm

    # Confidence proxy: based on ensemble agreement & feature completeness
    comp_preds = np.vstack([preds_components['rf'], preds_components['gb'], preds_components['ridge'], preds_components.get('nn', np.zeros(len(X)))])
    agreement = 1 - np.std(comp_preds, axis=0)  # higher std -> lower agreement, invert
    feature_quality = 1.0 - (X.isna().sum(axis=1) / len(X.columns))
    confidence = np.clip(0.6*agreement + 0.4*feature_quality.values, 0, 1)

    df_prep['ml_score'] = scores_norm
    df_prep['trad_score'] = trad_norm
    df_prep['score_final'] = final_score
    df_prep['confidence'] = confidence

    # Ranking
    df_ranked = df_prep.sort_values('score_final', ascending=False).reset_index(drop=True)
    df_ranked['rang'] = range(1, len(df_ranked)+1)

    # Explain approximate contributions
    if show_explain:
        imp_norm, row_top_feats = hrm.explain_contributions(X)
        df_ranked['top_feats'] = row_top_feats
    else:
        imp_norm = {}
        df_ranked['top_feats'] = [{} for _ in range(len(df_ranked))]

    # Display results
    st.markdown("## 🏆 Classement final")
    display_cols = ['rang','Nom','Cote','score_final','confidence']
    display_df = df_ranked[display_cols].copy()
    display_df['score_final'] = display_df['score_final'].apply(lambda x: f"{x:.3f}")
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    st.dataframe(display_df, use_container_width=True, height=350)

    # Top 5 with details
    st.markdown("### 🥇 Top 5 prédictions détaillées")
    for i in range(min(5, len(df_ranked))):
        r = df_ranked.iloc[i]
        st.markdown(f"**{i+1}. {r['Nom']}** — Cote: {r['Cote']} — Score: {r['score_final']:.3f} — Confiance: {r['confidence']:.1%}")
        if show_explain:
            st.markdown(f"Contribs approx: {r['top_feats']}")

    # Visual
    fig = create_summary_fig(df_ranked)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance display (approx from RF)
    try:
        fi = dict(zip(X.columns, hrm.rf.feature_importances_))
        fi_df = pd.DataFrame(fi.items(), columns=['Feature','Importance']).sort_values('Importance', ascending=False)
        st.subheader("🔬 Importance features (RandomForest - approx.)")
        st.dataframe(fi_df.head(15), use_container_width=True, height=250)
    except Exception:
        pass

    # Combinations
    st.markdown("### 🎲 Combinaisons suggérées")
    combos_trio = generate_combinations(df_ranked, 'trio', topk=6)
    if combos_trio:
        st.markdown("**Trio (sélections sans ordre) — quelques combinaisons**")
        for c in combos_trio[:10]:
            st.write(" - " + " | ".join(c))
    combos_quinte = generate_combinations(df_ranked, 'quinte', topk=6)
    if combos_quinte:
        st.markdown("**Quinté (top5)**")
        st.write(combos_quinte[0])

    # Exports
    st.markdown("---")
    st.subheader("💾 Exports")
    csv_data = df_ranked.to_csv(index=False)
    st.download_button("📄 Télécharger CSV", csv_data, f"pronostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    json_data = df_ranked.to_json(orient='records', indent=2)
    st.download_button("📋 Télécharger JSON", json_data, f"pronostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "application/json")

    # Report summary
    st.markdown("### 📊 Rapport synthétique")
    st.write(f"ML blending weights: {hrm.weights}")
    st.write(f"CV R2 (RF/GB): {hrm.cv_scores.get('rf'):.3f} / {hrm.cv_scores.get('gb'):.3f}" if hrm.cv_scores.get('rf') is not None else "CV non disponible")
    st.write("Feature weights (dynamic):")
    st.dataframe(pd.DataFrame(list(hrm.feature_weights.items()), columns=['Feature','Weight']).sort_values('Weight', ascending=False).head(20))

if __name__ == "__main__":
    main()
