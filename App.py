# analyseur_ml_geny_streamlit.py
"""
Streamlit app - Supervised Top-3 Predictor (Geny history)
- Train on historique_geny_auto.csv (or uploaded CSV)
- Predict probability to be placed (Top 3) for upcoming race (scrape Geny or upload race CSV)
- Uses GradientBoostingClassifier + MLPClassifier + LogisticRegression (Voting soft)
- Target encoding for jockey/trainer, correlation-based feature weighting
- No TensorFlow dependency (uses scikit-learn)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests, re, time
from bs4 import BeautifulSoup
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ML libs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.impute import SimpleImputer

# Plot
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Helper utils & scraping
# -------------------------
def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        s = str(x).strip().replace('%','').replace(',','.')
        return float(s)
    except:
        return default

def extract_weight(poids_str):
    if pd.isna(poids_str):
        return np.nan
    m = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
    return float(m.group(1).replace(',', '.')) if m else np.nan

def extract_music_positions(music_str):
    # Return list of integer positions found in the music string
    if not music_str or pd.isna(music_str): return []
    s = str(music_str)
    pos = [int(ch) for ch in s if ch.isdigit() and int(ch) > 0]
    return pos

def music_features_from_str(music_str):
    pos = extract_music_positions(music_str)
    if not pos:
        return {'m_wins':0,'m_places':0,'m_total':0,'m_win_rate':0.0,'m_place_rate':0.0,'m_recent':0.0,'m_avg_pos':10.0}
    total = len(pos)
    wins = pos.count(1)
    places = sum(1 for p in pos if p <= 3)
    win_rate = wins/total
    place_rate = places/total
    recent = pos[:3]
    recent_form = sum(1.0/p for p in recent)/len(recent) if recent else 0.0
    avg_pos = float(np.mean(pos))
    return {'m_wins':wins,'m_places':places,'m_total':total,'m_win_rate':win_rate,'m_place_rate':place_rate,'m_recent':recent_form,'m_avg_pos':avg_pos}

# Keep the earlier scrape_race_data (light)
def scrape_race_basic(url):
    """Scrapes a basic race participants table (names, cote, poids, musique, draw)
       This is a best-effort; adapt to the specific geny page structure you have.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; TurfBot/1.0)'}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find('table')
    if not table:
        return None, "No <table> found"
    rows = table.find_all('tr')[1:]
    rows_data = []
    for row in rows:
        cols = [c.get_text(strip=True) for c in row.find_all(['td','th'])]
        if len(cols) < 3:
            continue
        # heuristic mapping - adapt if need be
        num = cols[0]
        name = cols[1]
        music = cols[2] if len(cols) > 2 else ''
        # attempt weight and odds at last columns
        cote = cols[-1] if len(cols) >= 4 else ''
        poids = cols[-2] if len(cols) >= 5 else ''
        rows_data.append({'Num':num,'Nom':name,'Musique':music,'Cote':cote,'Poids':poids})
    if not rows_data:
        return None, "No rows parsed"
    return pd.DataFrame(rows_data), "ok"

# Scraper of "stats" pages (jockey/entraineur stats) - used when scraping history per course
def scrape_geny_stats_page(url):
    """Scrape the Geny stats page for one course (jockey/entraineur tables)"""
    headers = {'User-Agent':'Mozilla/5.0 (compatible; TurfBot/2.0)'}
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    soup = BeautifulSoup(r.content, 'html.parser')
    tables = soup.find_all('table')
    data = []
    for table in tables:
        header = table.find_previous('h3')
        if header and 'jockey' in header.get_text().lower():
            section = 'jockey'
        elif header and 'entraîneur' in header.get_text().lower():
            section = 'entraineur'
        else:
            continue
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all('td')]
            if len(cols) < 5:
                continue
            # often cols: N°, Cheval, SA, Poids, Détail, Jockey, Courses, Victoires, ... Musique link
            # We'll map robustly by positions if possible
            try:
                cheval = cols[1]
                poids = cols[2] if len(cols) > 2 else ''
                person = cols[3] if len(cols) > 3 else ''
                courses = safe_float(cols[4])
                victoires = safe_float(cols[5]) if len(cols) > 5 else np.nan
                places = safe_float(cols[6]) if len(cols) > 6 else np.nan
                pct = safe_float(cols[7]) if len(cols) > 7 else np.nan
                musique = cols[-1] if len(cols) > 0 else ''
                data.append({'Cheval':cheval,'Poids':poids,'Person':person,'Courses':courses,'Victoires':victoires,'Places':places,'%Reussite':pct,'Musique':musique,'Section':section})
            except Exception:
                continue
    if not data: return None
    df = pd.DataFrame(data)
    # pivot out jockey vs entraineur per Cheval
    jockey_df = df[df['Section']=='jockey'].drop(columns=['Section']).rename(columns={'Person':'Jockey','Courses':'C_J','Victoires':'V_J','Places':'P_J','%Reussite':'Pct_J','Musique':'Musique_J'})
    ent_df = df[df['Section']=='entraineur'].drop(columns=['Section']).rename(columns={'Person':'Entraineur','Courses':'C_E','Victoires':'V_E','Places':'P_E','%Reussite':'Pct_E','Musique':'Musique_E'})
    merged = pd.merge(jockey_df, ent_df, on='Cheval', how='outer')
    return merged

# -------------------------
# Historical dataset preparation
# -------------------------
def prepare_history(df_hist):
    """
    df_hist expected to contain at least:
      - Cheval, Jockey, Entraineur (or Person/Jockey fields), Musique, Poids, Cote (opt), Rang/Position (opt)
    Output: cleaned DF with target 'placed' (1 if Rang<=3), and engineered features
    """
    df = df_hist.copy()
    # Normalize column names
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    # If Rang/Position present -> create placed target
    if 'Rang' in df.columns:
        df['Position'] = df['Rang']
    if 'Position' in df.columns:
        df['placed'] = df['Position'].apply(lambda x: 1 if (not pd.isna(x) and float(x) <= 3) else 0)
    elif 'Result' in df.columns:
        # Result might be 1/0 indicating win - fallback (placed only if 1 and <=3 unavailable)
        df['placed'] = df['Result'].apply(lambda x: 1 if float(x)==1 else 0)
    else:
        # If no labels: cannot train supervised model
        st.warning("Le fichier historique ne contient pas de colonne 'Position'/'Rang'/'Result'. Impossible d'entraîner un modèle supervisé.")
        df['placed'] = np.nan

    # Music features
    m_feats = df.get('Musique', pd.Series(['']*len(df))).apply(music_features_from_str)
    m_df = pd.DataFrame(m_feats.tolist()).fillna(0)
    df = pd.concat([df.reset_index(drop=True), m_df.reset_index(drop=True)], axis=1)

    # numeric conversions
    if 'Poids' in df.columns:
        df['weight_kg'] = df['Poids'].apply(extract_weight)
    else:
        df['weight_kg'] = np.nan

    if 'Cote' in df.columns:
        df['odds_numeric'] = df['Cote'].apply(lambda x: safe_float(x, default=np.nan))
    else:
        df['odds_numeric'] = np.nan
    df['odds_inv'] = 1.0 / (df['odds_numeric'] + 0.1)
    # Age if present
    if 'Âge/Sexe' in df.columns:
        df['age'] = df['Âge/Sexe'].astype(str).str.extract(r'(\d+)').astype(float).fillna(np.nan)
    else:
        df['age'] = np.nan

    # Trim names
    for c in ['Jockey','Entraineur','Jockey_x','Entraineur_x','Person','Jockey_J','Entraineur_E']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # unify jockey/trainer names columns if multiple variants found
    # heuristic: take first matching column name
    if 'Jockey' not in df.columns:
        candidates = [c for c in df.columns if 'jockey' in c.lower()]
        if candidates: df['Jockey'] = df[candidates[0]]
    if 'Entraineur' not in df.columns:
        candidates = [c for c in df.columns if 'entraineur' in c.lower() or 'entra' in c.lower()]
        if candidates: df['Entraineur'] = df[candidates[0]]

    return df

# -------------------------
# Feature engineering & target encoding
# -------------------------
def build_features_and_encoders(df_train):
    """Compute encoding maps (target encoding for jockey/trainer) and return X, y, encoders"""
    df = df_train.copy()
    # drop rows without target
    df = df[~df['placed'].isna()].reset_index(drop=True)
    y = df['placed'].astype(int).values

    # Target encoding for Jockey and Entraineur: mean placed
    jockey_map = df.groupby('Jockey')['placed'].mean().to_dict() if 'Jockey' in df.columns else {}
    trainer_map = df.groupby('Entraineur')['placed'].mean().to_dict() if 'Entraineur' in df.columns else {}
    global_mean = float(df['placed'].mean())

    def map_jockey(x): return jockey_map.get(x, global_mean)
    def map_trainer(x): return trainer_map.get(x, global_mean)

    df['Jockey_te'] = df.get('Jockey','').apply(map_jockey)
    df['Trainer_te'] = df.get('Entraineur','').apply(map_trainer)

    # Basic features to use:
    feature_cols = []
    # numeric
    for c in ['weight_kg','age','odds_numeric','odds_inv','m_wins','m_places','m_total','m_win_rate','m_place_rate','m_recent','m_avg_pos']:
        if c in df.columns:
            feature_cols.append(c)
    # add target encodings and field size if present
    feature_cols += ['Jockey_te','Trainer_te']
    # if draw exists
    if 'Numéro de corde' in df.columns:
        df['draw'] = df['Numéro de corde'].apply(lambda x: safe_float(x, default=np.nan))
        feature_cols.append('draw')
    # fill na
    imp = SimpleImputer(strategy='median')
    X_raw = pd.DataFrame(imp.fit_transform(df[feature_cols]), columns=feature_cols)
    # scale
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols)

    # compute feature weights by correlation
    weights = compute_feature_weights(X_scaled, df['placed'].values)
    # apply weights as multiplicative emphasis
    X_weighted = X_scaled.copy()
    for c,w in weights.items():
        if c in X_weighted.columns:
            X_weighted[c] = X_weighted[c] * w

    encoders = {'jockey_map':jockey_map,'trainer_map':trainer_map,'global_mean':global_mean,'imputer':imp,'scaler':scaler,'weights':weights,'feature_cols':feature_cols}
    return X_weighted, y, encoders

def compute_feature_weights(X, y):
    """Pearson+Spearman absolute combined normalized to [0.2,1.0]"""
    import math
    from scipy.stats import spearmanr
    scores = {}
    arr_y = np.array(y, dtype=float)
    for c in X.columns:
        xv = np.array(X[c].fillna(0), dtype=float)
        # Pearson
        try:
            p = np.corrcoef(xv, arr_y)[0,1]
            if np.isnan(p): p = 0.0
        except:
            p = 0.0
        # Spearman
        try:
            s = spearmanr(xv, arr_y).correlation
            if np.isnan(s): s = 0.0
        except:
            s = 0.0
        score = 0.5*(abs(p)+abs(s))
        scores[c] = float(score)
    vals = np.array(list(scores.values()))
    if vals.max()==vals.min():
        return {k:1.0 for k in scores.keys()}
    mn, mx = vals.min(), vals.max()
    norm = {k: 0.2 + 0.8*((v-mn)/(mx-mn)) for k,v in scores.items()}  # floor 0.2
    return norm

# -------------------------
# Model training
# -------------------------
def train_model(X, y):
    """
    Train a VotingClassifier (soft) from X,y
    Returns calibrated voting model and cv score dict
    """
    # base learners
    gb = GradientBoostingClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', max_iter=1000, random_state=42)
    lr = LogisticRegression(max_iter=1000, solver='liblinear')

    ensemble = VotingClassifier(estimators=[('gb',gb),('mlp',mlp),('lr',lr)], voting='soft', n_jobs=-1)
    # Fit ensemble
    ensemble.fit(X, y)
    # Calibrate (isotonic or sigmoid; use sigmoid for stability)
    calib = CalibratedClassifierCV(base_estimator=ensemble, method='sigmoid', cv='prefit')
    calib.fit(X, y)
    # cross-validate AUC
    try:
        cv_auc = np.mean(cross_val_score(calib, X, y, cv=5, scoring='roc_auc', n_jobs=-1))
    except Exception:
        cv_auc = None

    return calib, {'cv_auc':cv_auc}

# -------------------------
# Application to new race
# -------------------------
def prepare_race_features(df_race, encoders):
    """
    df_race: dataframe of participants with columns 'Nom','Musique','Poids','Cote','Numéro de corde','Jockey','Entraineur' ideally
    encoders: maps from training (jockey_map, trainer_map, imputer, scaler, weights, feature_cols)
    """
    df = df_race.copy()
    # music features
    m_feats = df.get('Musique', pd.Series(['']*len(df))).apply(music_features_from_str)
    m_df = pd.DataFrame(m_feats.tolist()).fillna(0)
    df = pd.concat([df.reset_index(drop=True), m_df.reset_index(drop=True)], axis=1)

    # numeric conversions
    df['weight_kg'] = df.get('Poids', '').apply(extract_weight) if 'Poids' in df.columns else np.nan
    df['odds_numeric'] = df.get('Cote', np.nan).apply(lambda x: safe_float(x, default=np.nan)) if 'Cote' in df.columns else np.nan
    df['odds_inv'] = 1.0/(df['odds_numeric']+0.1)
    if 'Numéro de corde' in df.columns:
        df['draw'] = df['Numéro de corde'].apply(lambda x: safe_float(x, default=np.nan))
    # map target encodings
    jm = encoders['jockey_map']
    tm = encoders['trainer_map']
    gm = encoders['global_mean']
    df['Jockey_te'] = df.get('Jockey','').apply(lambda x: jm.get(x, gm))
    df['Trainer_te'] = df.get('Entraineur','').apply(lambda x: tm.get(x, gm))
    # build feature dataframe in same order
    feat_cols = encoders['feature_cols']
    # ensure columns exist
    for c in feat_cols:
        if c not in df.columns:
            df[c] = np.nan
    # impute and scale
    imp = encoders['imputer']
    scaler = encoders['scaler']
    X_raw = pd.DataFrame(imp.transform(df[feat_cols]), columns=feat_cols)
    X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=feat_cols)
    # apply weights
    weights = encoders['weights']
    X_weighted = X_scaled.copy()
    for c,w in weights.items():
        if c in X_weighted.columns:
            X_weighted[c] = X_weighted[c] * w
    return X_weighted

# -------------------------
# Main Streamlit app
# -------------------------
st.set_page_config(page_title="Analyseur ML Geny - Place Top3", layout="wide")
st.title("🏇 Analyseur ML — Probabilité de place (Top 3)")

st.markdown("""
Ce dashboard entraîne un modèle supervisé sur ton historique Geny (fichier CSV) et prédit la probabilité qu'un cheval **soit placé (Top 3)**.
- Étape 1 : charger / générer historique (scraper Geny fourni séparément)
- Étape 2 : entraîner le modèle
- Étape 3 : appliquer sur une course à venir (URL Geny ou upload CSV)
""")

# Sidebar: load historical dataset
st.sidebar.header("1) Données historiques")
hist_mode = st.sidebar.selectbox("Source historique", ["Charger CSV", "Charger automatique (fichier existant historique_geny_auto.csv)"])
if hist_mode == "Charger CSV":
    uploaded_hist = st.sidebar.file_uploader("Upload historique CSV (contenant Rang/Position)", type=['csv'])
    if uploaded_hist is not None:
        df_hist = pd.read_csv(uploaded_hist)
    else:
        df_hist = None
else:
    # attempt to load default file
    try:
        df_hist = pd.read_csv("historique_geny_auto.csv")
        st.sidebar.success("historique_geny_auto.csv chargé depuis le dossier courant")
    except Exception:
        df_hist = None
        st.sidebar.warning("historique_geny_auto.csv introuvable. Uploadez un CSV ou exécutez le scraper.")

if df_hist is None:
    st.info("Téléverse un fichier historique (CSV) ou place 'historique_geny_auto.csv' dans le dossier.")
    st.stop()

st.subheader("Aperçu historique (quelques lignes)")
st.dataframe(df_hist.head(10))

# Prepare history
with st.spinner("Préparation des données historiques..."):
    df_hist_prep = prepare_history(df_hist)

# If no targets, stop
if df_hist_prep['placed'].isna().all():
    st.error("Le dataset historique ne contient aucune étiquette 'placed'. Impossible d'entraîner.")
    st.stop()

# Build features and encoders
X_weighted, y, encoders = build_features_and_encoders(df_hist_prep)

st.markdown("### Stats d'entraînement")
st.write(f"Nombre d'exemples: {len(X_weighted)} ; % placé: {np.mean(y):.2%}")

# Train model
if st.button("▶️ Entraîner modèle maintenant"):
    with st.spinner("Entraînement du modèle (cela peut prendre quelques dizaines de secondes)..."):
        model, metrics = train_model(X_weighted, y)
    st.success("Modèle entraîné et calibré")
    if metrics.get('cv_auc') is not None:
        st.write(f"CV AUC (approx) : {metrics['cv_auc']:.3f}")
    else:
        st.write("CV AUC non disponible")

    # Save model and encoders in session state
    st.session_state['model'] = model
    st.session_state['encoders'] = encoders
else:
    model = st.session_state.get('model', None)
    encoders = st.session_state.get('encoders', encoders)

if model is None:
    st.info("Entraîne le modèle pour pouvoir l'appliquer sur une course.")
    st.stop()

# -----------------------
# Apply model to a race
# -----------------------
st.header("2) Appliquer sur une course à venir")
apply_mode = st.selectbox("Mode d'entrée course", ["Scrape URL Geny", "Upload CSV course (Nom,Cote,Poids,Musique,Jockey,Entraineur,Numéro de corde)"])
df_race = None
if apply_mode == "Scrape URL Geny":
    race_url = st.text_input("URL course Geny (page participants) :", "")
    if st.button("Scraper et analyser cette course") and race_url:
        with st.spinner("Scraping course..."):
            df_race, msg = scrape_race_basic(race_url)
            if df_race is None:
                st.error(f"Erreur scraping: {msg}")
            else:
                st.success("Course scrappée")
                # optional: let user input Jockey/Entraineur mapping if missing
                st.write("Aperçu participants :")
                st.dataframe(df_race)
elif apply_mode == "Upload CSV course (Nom,Cote,Poids,Musique,Jockey,Entraineur,Numéro de corde)":
    uploaded_course = st.file_uploader("Upload CSV course", type=['csv'])
    if uploaded_course is not None:
        df_race = pd.read_csv(uploaded_course)
        st.success("Course uploadée")
        st.dataframe(df_race.head())

if df_race is None:
    st.info("Scrapez ou uploadez une course pour obtenir les pronostics.")
    st.stop()

# Prepare race features
with st.spinner("Préparation features de la course..."):
    X_race = prepare_race_features(df_race, encoders)

# Predict probabilities
with st.spinner("Prédiction..."):
    proba = model.predict_proba(X_race)[:,1]  # probability placed
    df_race = df_race.reset_index(drop=True)
    df_race['proba_place'] = proba
    # Implied probability from odds
    if 'Cote' in df_race.columns:
        df_race['odds_numeric'] = df_race['Cote'].apply(lambda x: safe_float(x, default=np.nan))
        # implied probability approx = 1 / odds (not perfectly accurate but ok)
        df_race['implied_proba'] = 1.0 / (df_race['odds_numeric'] + 1e-6)
    else:
        df_race['implied_proba'] = np.nan

    # Value score
    df_race['value_ratio'] = df_race['proba_place'] / (df_race['implied_proba'] + 1e-9)

    # Final ranking
    df_ranked = df_race.sort_values('proba_place', ascending=False).reset_index(drop=True)
    df_ranked['rank'] = df_ranked.index + 1

# Display predictions
st.subheader("Résultats - Probabilité de place (Top 3)")
display_cols = ['rank','Num','Nom','Cote','proba_place','implied_proba','value_ratio','Poids','Musique','Jockey','Entraineur']
present_cols = [c for c in display_cols if c in df_ranked.columns]
df_display = df_ranked[present_cols].copy()
if 'proba_place' in df_display.columns:
    df_display['proba_place'] = df_display['proba_place'].apply(lambda x: f"{x:.2%}")
if 'implied_proba' in df_display.columns:
    df_display['implied_proba'] = df_display['implied_proba'].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "")
if 'value_ratio' in df_display.columns:
    df_display['value_ratio'] = df_display['value_ratio'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "")

st.dataframe(df_display, use_container_width=True)

# Top suggestions by value
st.subheader("Chevaux avec 'Edge' (value_ratio élevé)")
if 'value_ratio' in df_ranked.columns:
    vr = df_ranked.dropna(subset=['value_ratio']).sort_values('value_ratio', ascending=False)
    if len(vr) > 0:
        st.table(vr[['Nom','Cote','proba_place','implied_proba','value_ratio']].head(10).assign(proba_place=lambda d: d['proba_place'].apply(lambda x: f"{x:.2%}"), implied_proba=lambda d: d['implied_proba'].apply(lambda x: f"{x:.2%}" if not pd.isna(x) else "")))
    else:
        st.info("Aucune cote trouvée pour calculer value_ratio.")
else:
    st.info("Value ratio indisponible : pas de colonne 'Cote' fournie.")

# Visuals
st.subheader("Visualisations")
fig = px.bar(df_ranked, x='Nom', y='proba_place', title="Probabilité estimée de place (Top 3)", text=df_ranked['proba_place'].apply(lambda x: f"{x:.1%}"))
st.plotly_chart(fig, use_container_width=True)

# Feature importance (approx) from training GB
try:
    gb = model.base_estimator_.estimators_[0] if hasattr(model.base_estimator_, 'estimators_') else None
    # if we used calibrated classifier, the base_estimator_ is named; adapt
except Exception:
    gb = None

if hasattr(model, 'base_estimator_'):
    # try to get underlying estimators importances if possible
    st.subheader("Importance approximative des features (sur GB si disponible)")
    try:
        # In our pipeline, model might be CalibratedClassifierCV wrapping VotingClassifier
        base = model.base_estimator_  # VotingClassifier
        # get first estimator (gb) from VotingClassifier
        if hasattr(base, 'estimators_'):
            est_names = [n for n,_ in base.estimators]
            # try to access gradient boosting
            for n,e in base.estimators:
                if 'gb' in n:
                    gb_est = e
                    break
            else:
                gb_est = None
        else:
            gb_est = None
        if gb_est is not None and hasattr(gb_est, 'feature_importances_'):
            fi = dict(zip(encoders['feature_cols'], gb_est.feature_importances_))
            fi_df = pd.DataFrame(list(fi.items()), columns=['feature','importance']).sort_values('importance', ascending=False).head(20)
            st.dataframe(fi_df, use_container_width=True)
            fig2 = px.bar(fi_df, x='feature', y='importance', title='Feature importances (GB)')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Feature importances non disponibles (GB non trouvé ou non exposé).")
    except Exception as e:
        st.write("Erreur extraction importances:", e)

# Export options
st.subheader("Export")
csv_out = df_ranked.to_csv(index=False)
st.download_button("Télécharger pronostics (CSV)", csv_out, file_name=f"pronostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
st.download_button("Télécharger pronostics (JSON)", df_ranked.to_json(orient='records', indent=2), file_name=f"pronostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

st.markdown("---")
st.markdown("Notes et recommandations :\n- Plus l'historique contient de courses et de résultats réels (Rang/Position), mieux le modèle s'entraînera.\n- Les cotes sont gardées mais peu pondérées ; le modèle apprend à partir des performances passées.\n- Pour une meilleure robustesse, collecter 1000+ lignes historiques (plusieurs courses) et réentraîner régulièrement.")
