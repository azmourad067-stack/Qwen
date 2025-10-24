# analyseur_hippique_pro_auto.py
# -*- coding: utf-8 -*-
"""
Analyseur Hippique PRO — version autonome & robuste
- Auto-entraînement incrémental (fusion historique)
- Modèle hybride: Deep Learning (Keras) + XGBoost (ensemble)
- Cross-validation (simple) pour ML, EarlyStopping pour DL
- Génération e-trio pondérée par probabilités, filtrage des combinaisons
- Simulation de gains, journalisation (logs/training_log.csv, data/historique.csv)
"""
import os
import re
from datetime import datetime
import json
import math

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import Sequential, callbacks
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# --- Directories & Paths ---
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

HIST_PATH = os.path.join("data", "historique.csv")
DL_MODEL_PATH = os.path.join("models", "dl_model.keras")
SCALER_PATH = os.path.join("models", "scaler.joblib")
XGB_PATH = os.path.join("models", "xgb_model.joblib")
TRAIN_LOG = os.path.join("logs", "training_log.csv")
PERF_LOG = os.path.join("logs", "performance_log.csv")

# --- Utilities ---
def safe_float(x, default=np.nan):
    try:
        if pd.isna(x): return default
        s = str(x).strip().replace(",", ".")
        return float(re.findall(r"-?\\d+(?:\\.\\d+)?", s)[0])
    except Exception:
        return default

def extract_weight(s):
    try:
        if pd.isna(s): return 60.0
        m = re.search(r"(\\d+(?:[.,]\\d+)?)", str(s))
        return float(m.group(1).replace(",", ".")) if m else 60.0
    except:
        return 60.0

# --- Scraper simple (table-based) ---
@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=12)
        if r.status_code != 200: return None, f"HTTP {r.status_code}"
        soup = BeautifulSoup(r.content, "html.parser")
        table = soup.find("table")
        if not table: return None, "Aucun tableau trouvé"
        rows = table.find_all("tr")[1:]
        data=[]
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all(["td","th"])]
            if len(cols) < 3: continue
            # best-effort mapping (user CSV preferred)
            data.append({
                "Numéro de corde": cols[0] if len(cols)>0 else "",
                "Nom": cols[1] if len(cols)>1 else "",
                "Musique": cols[2] if len(cols)>2 else "",
                "Poids": cols[-2] if len(cols)>3 else "60",
                "Cote": cols[-1]
            })
        if not data: return None, "Aucune donnée extraite"
        return pd.DataFrame(data), "Succès"
    except Exception as e:
        return None, f"Erreur: {e}"

# --- Feature engineering (robuste) ---
def music_to_features(music):
    s = str(music)
    digits = [int(x) for x in re.findall(r"\\d+", s)]
    if not digits:
        return {"recent_wins":0, "recent_top3":0, "weighted":0.0}
    recent_wins = sum(1 for d in digits if d==1)
    recent_top3 = sum(1 for d in digits if d<=3)
    weights = np.linspace(1.0, 0.3, num=len(digits))
    weighted = sum((4 - d) * w for d,w in zip(digits, weights)) / (len(digits)+1e-6)
    return {"recent_wins":recent_wins, "recent_top3":recent_top3, "weighted":weighted}

def prepare_data(df):
    df = df.copy()
    # unify columns presence
    for col in ["Cote","Numéro de corde","Poids","Musique","Âge/Sexe","Nom"]:
        if col not in df.columns:
            df[col] = ""
    df["odds_numeric"] = df["Cote"].apply(lambda x: safe_float(x, default=np.nan)).fillna(999.0)
    df["draw_numeric"] = df["Numéro de corde"].apply(lambda x: safe_float(x, default=1)).fillna(1).astype(int)
    df["weight_kg"] = df["Poids"].apply(extract_weight)
    ages=[]
    is_female=[]
    rw=[]; rt3=[]; rwght=[]
    for a in df.get("Âge/Sexe", [""]*len(df)):
        m = re.search(r"(\\d+)", str(a))
        ages.append(float(m.group(1)) if m else 4.0)
        s = str(a).upper()
        is_female.append(1 if "F" in s else 0)
    for m in df.get("Musique", [""]*len(df)):
        feat = music_to_features(m)
        rw.append(feat["recent_wins"]); rt3.append(feat["recent_top3"]); rwght.append(feat["weighted"])
    df["age"] = ages
    df["is_female"] = is_female
    df["recent_wins"] = rw
    df["recent_top3"] = rt3
    df["recent_weighted"] = rwght
    # optional jockey/entraîneur features if present (pass-through)
    if "Jockey" not in df.columns: df["Jockey"] = ""
    if "Entraineur" not in df.columns: df["Entraineur"] = ""
    # keep only valid odds
    df = df[df["odds_numeric"] > 0]
    df = df.reset_index(drop=True)
    return df

# --- History management (fusion automatique) ---
def append_to_historique(df_new):
    """
    Append new raw race dataframe to historique.csv.
    We keep raw columns so we can re-extract features later.
    """
    try:
        df_copy = df_new.copy()
        df_copy["source_ts"] = datetime.now().isoformat()
        if os.path.exists(HIST_PATH):
            old = pd.read_csv(HIST_PATH)
            combined = pd.concat([old, df_copy], ignore_index=True)
        else:
            combined = df_copy
        combined.to_csv(HIST_PATH, index=False)
        return True, len(combined)
    except Exception as e:
        return False, str(e)

# --- Model manager Hybrid (DL + XGBoost) ---
class HybridModel:
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or ["odds_numeric","draw_numeric","weight_kg","age","is_female","recent_wins","recent_top3","recent_weighted"]
        self.scaler = StandardScaler()
        self.dl_model = None
        self.xgb_model = None
        self.loaded = False
        # try load persisted
        if os.path.exists(DL_MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.dl_model = tf.keras.models.load_model(DL_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                if os.path.exists(XGB_PATH):
                    self.xgb_model = joblib.load(XGB_PATH)
                self.loaded = True
            except Exception as e:
                st.warning(f"Chargement modèles: {e}")

    def build_dl(self, input_dim):
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

    def train(self, X, y, dl_epochs=20, dl_batch=8, xgb_rounds=100, val_split=0.15, use_cv=False):
        """
        Train DL and XGB on X,y (numpy arrays or pd.DataFrame/Series).
        If historique is available with many rows, we can do CV for xgb.
        """
        # X may be numpy or df
        X_np = X.values if hasattr(X, "values") else np.array(X)
        y_np = y.values if hasattr(y, "values") else np.array(y)
        # scale
        Xs = self.scaler.fit_transform(X_np)
        # DL
        if self.dl_model is None:
            self.dl_model = self.build_dl(Xs.shape[1])
        es = callbacks.EarlyStopping(patience=6, restore_best_weights=True)
        history = self.dl_model.fit(Xs, y_np, validation_split=val_split, epochs=dl_epochs, batch_size=dl_batch, callbacks=[es], verbose=0)
        # XGBoost training with basic CV if requested
        try:
            if use_cv and len(Xs) >= 50:
                # quick CV to choose rounds
                dtrain = xgb.DMatrix(Xs, label=y_np)
                params = {"objective":"reg:squarederror","learning_rate":0.05, "max_depth":4, "eval_metric":"rmse"}
                cvres = xgb.cv(params, dtrain, num_boost_round=xgb_rounds, nfold=4, early_stopping_rounds=10, verbose_eval=False)
                best_rounds = len(cvres)
            else:
                best_rounds = max(10, min(200, xgb_rounds))
            self.xgb_model = xgb.XGBRegressor(n_estimators=best_rounds, learning_rate=0.05, max_depth=4, random_state=42)
            self.xgb_model.fit(Xs, y_np, verbose=False)
        except Exception as e:
            st.warning(f"Erreur XGBoost training: {e}")
            self.xgb_model = None
        # persist
        try:
            self.dl_model.save(DL_MODEL_PATH, overwrite=True)
            joblib.dump(self.scaler, SCALER_PATH)
            if self.xgb_model is not None:
                joblib.dump(self.xgb_model, XGB_PATH)
        except Exception as e:
            st.warning(f"Erreur sauvegarde modèles: {e}")
        # log training summary
        loss = history.history.get("loss", [None])[-1]
        with open(TRAIN_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()},{len(Xs)},{loss},{DL_MODEL_PATH}\\n")
        return history.history

    def predict(self, X_df):
        """
        Returns:
            preds_ensemble, preds_dl, preds_xgb
        """
        if len(X_df)==0:
            return np.array([]), np.array([]), np.array([])
        X_np = X_df.values if hasattr(X_df, "values") else np.array(X_df)
        Xs = self.scaler.transform(X_np)
        preds_dl = self.dl_model.predict(Xs).flatten() if self.dl_model is not None else np.zeros(len(Xs))
        preds_xgb = self.xgb_model.predict(Xs) if self.xgb_model is not None else np.zeros(len(Xs))
        # normalize each
        def norm(a):
            a = np.array(a, dtype=float)
            if a.max() != a.min():
                return (a - a.min())/(a.max()-a.min())
            return np.zeros_like(a)
        ndl = norm(preds_dl)
        nx = norm(preds_xgb)
        # ensemble: weighted average (DL 60% XGB 40%)
        if self.xgb_model is not None:
            ensemble = 0.6*ndl + 0.4*nx
        else:
            ensemble = ndl
        return ensemble, ndl, nx

# --- Scoring helpers & confidence ---
def compute_confidence(dl_preds, xgb_preds):
    # confidence based on agreement (low std -> high confidence)
    arr = np.vstack([dl_preds, xgb_preds]) if len(xgb_preds)>0 else np.vstack([dl_preds])
    std = np.std(arr, axis=0)
    # map std -> confidence in (0..1) inverted
    conf = 1.0 - (std / (std.max()+1e-9))
    conf = np.clip(conf, 0.0, 1.0)
    return conf

# --- Combination generator weighted + filters ---
def generate_weighted_trios(df_ranked, n=35, max_total_odds=100.0, avoid_same_trainer=True):
    """
    - df_ranked: dataframe with 'Nom','score_final','Cote','Entraineur' (optional)
    - generate n distinct combos weighted by score_final probabilities
    - apply filters: remove combos with sum of cotes too large, or same trainer repeated
    """
    names = df_ranked["Nom"].tolist()
    scores = df_ranked["score_final"].values
    # ensure positive
    scores = np.maximum(scores, 1e-6)
    probs = scores / scores.sum()
    combos_set = set()
    results=[]
    attempts=0
    max_attempts = max(10000, n*200)
    while len(results) < n and attempts < max_attempts:
        attempts += 1
        chosen = np.random.choice(names, size=3, replace=False, p=probs)
        key = tuple(sorted(chosen))
        if key in combos_set: 
            continue
        # filters
        sub = df_ranked[df_ranked["Nom"].isin(chosen)]
        # total odds check
        try:
            total_odds = sub["odds_numeric"].astype(float).sum()
            if total_odds > max_total_odds: 
                continue
        except:
            pass
        # same trainer filter
        if avoid_same_trainer and "Entraineur" in sub.columns:
            trainers = sub["Entraineur"].fillna("").tolist()
            if len(set(trainers)) < len(trainers):
                continue
        combos_set.add(key)
        results.append(tuple(chosen))
    return results

# --- Simulation: simple backtest by applying combinations to historical results (if available) ---
def simulate_simple_returns(historique_df, combos_list, bet_amount=1.0, payoff_multiplier=lambda odds: 1/odds):
    """
    Very simple simulator: for each historic race, check if any combo matches the real top3 (if we have placements).
    This function expects historique_df to contain results (columns 'placement' or 'rank' or 'ResultTop3' etc.)
    Because historical structure is inconsistent, this simulator will only run if we detect 'placement' column.
    """
    # This is a placeholder to encourage storing actual results in historique.
    # Real simulation requires consistent labels (e.g. final positions).
    return {"simulated_return": 0.0, "notes": "Simulation requires labelled historical results (placements)."}

# --- Streamlit App UI & flow ---
st.set_page_config(page_title="🏇 Analyseur Hippique PRO", page_icon="🏇", layout="wide")
st.title("🏇 Analyseur Hippique PRO — Auto DL + Ensemble")

with st.sidebar:
    st.header("⚙️ Configuration")
    auto_dl = st.checkbox("Activer Auto DL & Auto-train", value=True)
    dl_epochs = st.number_input("DL epochs (per auto-train)", min_value=2, max_value=500, value=16, step=2)
    dl_batch = st.number_input("DL batch size", min_value=2, max_value=64, value=8, step=1)
    use_cv = st.checkbox("Utiliser CV pour XGBoost (si historique >= 50)", value=True)
    ml_confidence = st.slider("Poids ML/DL dans mélange final (0=heuristique seul, 1=modèles)", 0.0, 1.0, 0.6, 0.05)
    n_combos = st.number_input("Nb combinaisons e-trio", min_value=5, max_value=200, value=35, step=1)
    st.markdown("---")
    st.info("L'app fusionne automatiquement les nouveaux CSV/URLs dans data/historique.csv et réentraîne les modèles si auto DL est activé.")

tab1, tab2, tab3 = st.tabs(["🌐 URL Analysis","📁 Upload CSV","🧪 Test Data / Historique"])
df_race = None

with tab1:
    st.subheader("🔍 Analyse d'URL de course")
    url = st.text_input("URL de la page course (table HTML)")
    if st.button("Analyser URL"):
        if not url:
            st.error("Fournis une URL")
        else:
            df_tmp, msg = scrape_race_data(url)
            if df_tmp is None:
                st.error(f"Erreur: {msg}")
            else:
                st.success(f"✅ {len(df_tmp)} chevaux extraits")
                st.dataframe(df_tmp.head())
                df_race = df_tmp

with tab2:
    st.subheader("📤 Upload CSV (nouvelle course ou lot de courses)")
    uploaded = st.file_uploader("Fichier CSV (colonnes attendues: Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe, optional: Jockey, Entraineur)", type=["csv"])
    if uploaded:
        try:
            df_race = pd.read_csv(uploaded)
            st.success(f"✅ {len(df_race)} lignes chargées")
            st.dataframe(df_race.head())
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")

with tab3:
    st.subheader("🧪 Données de test & Historique")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Charger sample plat"):
            df_race = pd.DataFrame({
                "Nom": ["Thunder","Lightning","Storm","Rain","Wind"],
                "Numéro de corde": ["1","2","3","4","5"],
                "Cote": ["3.2","4.8","7.5","6.2","9.1"],
                "Poids": ["56.5","57","58.5","59","57.5"],
                "Musique": ["1a2a3a","2a1a4a","3a3a1a","1a4a2a","4a2a5a"],
                "Âge/Sexe": ["4H","5M","3F","6H","4M"]
            })
            st.success("Sample chargé")
            st.dataframe(df_race)
    with col2:
        if st.button("Afficher historique (si existe)"):
            if os.path.exists(HIST_PATH):
                hist = pd.read_csv(HIST_PATH)
                st.write(f"Historique {len(hist)} lignes")
                st.dataframe(hist.tail(50))
            else:
                st.info("Aucun historique trouvé")

# --- Main pipeline ---
if df_race is not None and len(df_race)>0:
    st.markdown("---")
    st.header("🔎 Préparation & Auto-entraînement")
    st.write("Extraction des features...")
    df_prepared = prepare_data(df_race)
    st.dataframe(df_prepared[["Nom","Cote","odds_numeric","draw_numeric","weight_kg","recent_wins","recent_top3","recent_weighted"]].assign(score=lambda d: (1/(d["odds_numeric"]+0.1)).round(3)))
    # append to historique
    ok, info = append_to_historique(df_race)
    if ok:
        st.success(f"Historique mis à jour ({info} lignes totales).")
    else:
        st.warning(f"Historique NON mis à jour: {info}")

    # instantiate hybrid model manager
    feats = ["odds_numeric","draw_numeric","weight_kg","age","is_female","recent_wins","recent_top3","recent_weighted"]
    model = HybridModel(feature_cols=feats)

    # prepare X/y from historique if available
    X_hist, y_hist = None, None
    if os.path.exists(HIST_PATH):
        try:
            hist_df = pd.read_csv(HIST_PATH)
            hist_prep = prepare_data(hist_df)
            # only include races that have 'placement' or 'rank' if possible
            if "placement" in hist_df.columns or "rank" in hist_df.columns:
                if "placement" in hist_df.columns:
                    y_hist = 1.0/(hist_prep["placement"].astype(float) + 0.1)
                else:
                    y_hist = 1.0/(hist_prep["rank"].astype(float) + 0.1)
                X_hist = hist_prep[feats]
            else:
                # pseudo-target fallback
                y_hist = 0.7*(1.0/(hist_prep["odds_numeric"]+0.1)) + 0.3*(hist_prep["recent_weighted"]/(hist_prep["recent_weighted"].max()+1e-6))
                X_hist = hist_prep[feats]
        except Exception as e:
            st.warning(f"Erreur lecture historique pour training: {e}")

    # Auto-train if enabled (prefer full historique)
    trained = False
    if auto_dl:
        try:
            if X_hist is not None and len(X_hist) >= 4:
                st.info(f"Entrainement automatique sur historique ({len(X_hist)} échantillons)...")
                hist = model.train(X_hist, y_hist, dl_epochs=dl_epochs, dl_batch=dl_batch, use_cv=use_cv)
                trained = True
            else:
                # train on current race only (pseudo-target)
                if len(df_prepared) >= 3:
                    y_curr = 0.7*(1.0/(df_prepared["odds_numeric"]+0.1)) + 0.3*(df_prepared["recent_weighted"]/(df_prepared["recent_weighted"].max()+1e-6))
                    st.info("Peu de données historiques: entraînement sur la course actuelle (pseudo-target).")
                    hist = model.train(df_prepared[feats], y_curr, dl_epochs=max(4,dl_epochs//2), dl_batch=dl_batch, use_cv=False)
                    trained = True
        except Exception as e:
            st.warning(f"Erreur auto-train: {e}")

    # predictions
    st.info("Calcul des prédictions hybrid (DL + XGBoost)...")
    X_curr = df_prepared[feats].fillna(0)
    ensemble, dl_p, xgb_p = model.predict(X_curr)
    # compute a fallback heuristic score
    heuristic = 1.0/(df_prepared["odds_numeric"]+0.1)
    if heuristic.max() != heuristic.min():
        heuristic = (heuristic - heuristic.min())/(heuristic.max()-heuristic.min())
    # blend final: (1-ml_confidence)*heuristic + ml_confidence*(ensemble)
    final = (1 - ml_confidence) * heuristic + ml_confidence * ensemble
    # compute confidence
    confidence = compute_confidence(dl_p, xgb_p) if xgb_p.size>0 else np.ones_like(dl_p)
    df_prepared["dl_norm"] = dl_p
    df_prepared["xgb_norm"] = xgb_p if len(xgb_p)==len(dl_p) else np.zeros_like(dl_p)
    df_prepared["score_final"] = final
    df_prepared["confidence"] = confidence
    df_ranked = df_prepared.sort_values("score_final", ascending=False).reset_index(drop=True)
    df_ranked["rang"] = range(1, len(df_ranked)+1)

    # Display ranking & metrics
    left, right = st.columns([2,1])
    with left:
        st.subheader("🏆 Classement final")
        display_cols = ["rang","Nom","Cote","Numéro de corde","Poids","score_final","confidence"]
        display = df_ranked[display_cols].copy()
        display["Score"] = display["score_final"].round(3)
        display["Conf"] = (display["confidence"]*100).round(1).astype(str) + "%"
        display = display.drop(["score_final","confidence"], axis=1)
        st.dataframe(display, use_container_width=True)
    with right:
        st.subheader("📊 Métriques & infos")
        st.metric("Nb chevaux", len(df_ranked))
        # show training status
        st.markdown(f"- Auto-DL: **{'ON' if auto_dl else 'OFF'}**")
        st.markdown(f"- Modèles chargés: DL={'Oui' if model.dl_model is not None else 'Non'}, XGB={'Oui' if model.xgb_model is not None else 'Non'}")
        top_fav = len(df_ranked[df_ranked["odds_numeric"]<5])
        st.markdown(f"- Favoris (<5): **{top_fav}**")
        if trained:
            st.success("✅ Auto-train effectué")
        else:
            st.info("ℹ️ Pas d'entraînement (données insuffisantes ou DL désactivé)")

    # Visuals
    st.subheader("📊 Visualisations")
    st.bar_chart(df_ranked.set_index("Nom")["score_final"].head(12))

    # Generate weighted combos with filtering
    st.subheader("🎲 Générateur e-trio pondéré & filtré")
    combos = generate_weighted_trios(df_ranked, n=int(n_combos), max_total_odds=120.0, avoid_same_trainer=True)
    if combos:
        for i,c in enumerate(combos):
            st.markdown(f"{i+1}. {c[0]} — {c[1]} — {c[2]}")
    else:
        st.info("Aucune combinaison générée (essayez d'augmenter max_total_odds ou vérifier la course).")

    # Simple simulated return (placeholder)
    st.subheader("📈 Simulation rapide (placeholder)")
    sim = simulate_simple_returns(None, combos)
    st.write(sim.get("notes"))

    # Export results
    st.subheader("💾 Export")
    csv_data = df_ranked.to_csv(index=False)
    st.download_button("Télécharger CSV pronostic", csv_data, f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    st.download_button("Télécharger JSON pronostic", df_ranked.to_json(orient="records", indent=2), f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    # Save last predictions snapshot
    try:
        snapshot_path = os.path.join("data", f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_ranked.to_csv(snapshot_path, index=False)
        st.success(f"Snapshot sauvegardé: {snapshot_path}")
    except Exception as e:
        st.warning(f"Erreur sauvegarde snapshot: {e}")

st.markdown("---")
st.markdown("Notes: 1) Pour obtenir des pronostics réellement performants il faut un historique labellisé (placements) de plusieurs centaines à milliers de courses. 2) Je peux t'aider à ajouter l'extraction de données publiques (Geny) et pipelines ETL pour consolider l'historique automatiquement.")
