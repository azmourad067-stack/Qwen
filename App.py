# analyseur_hippique_final.py
# -*- coding: utf-8 -*-
"""
Analyseur Hippique Final — Scraper Geny robuste + Auto DL (hybride) + Streamlit UI
"""

import os, re, math, json, warnings
from datetime import datetime
from itertools import combinations
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# try imports for ML/DL
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

# ------------------- Paths -------------------
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

HIST_PATH = os.path.join("data", "historique.csv")
DL_MODEL_PATH = os.path.join("models", "dl_model.keras")
SCALER_PATH = os.path.join("models", "scaler.joblib")
XGB_PATH = os.path.join("models", "xgb_model.joblib")
TRAIN_LOG = os.path.join("logs", "training_log.csv")

# ------------------- Utilities -------------------
def safe_float(x, default=np.nan):
    try:
        if pd.isna(x): return default
        s = str(x).strip().replace("\xa0", "").replace(",", ".")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group(0)) if m else default
    except Exception:
        return default

def extract_weight(s):
    try:
        if pd.isna(s): return 60.0
        m = re.search(r"(\d+(?:[.,]\d+)?)", str(s))
        return float(m.group(1).replace(",", ".")) if m else 60.0
    except:
        return 60.0

def clean_text(s):
    if pd.isna(s): return ""
    return re.sub(r"\s+", " ", str(s)).strip()

# ------------------- Robust scraper for Geny (and generic table) -------------------
def scrape_geny_course(url, timeout=12):
    """
    Scrape a race page (Geny-like). Returns pandas DataFrame with columns:
    Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe, Jockey, Entraîneur, Gains
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    # Geny pages often encoded ISO-8859-1
    if r.encoding is None or "utf" not in r.encoding.lower():
        r.encoding = "ISO-8859-1"
    soup = BeautifulSoup(r.text, "lxml")

    # Heuristics: find table with header containing 'Cheval' or 'Musique' or 'Gains'
    candidate_tables = soup.find_all("table")
    table = None
    for t in candidate_tables:
        ths = [th.get_text(strip=True).lower() for th in t.find_all(["th","td"])[:20]]
        joined = " ".join(ths)
        if any(k in joined for k in ["musique", "gains", "cheval", "rapports", "driver", "entraîneur", "entraineur"]):
            table = t
            break
    if table is None:
        # fallback to first table
        table = candidate_tables[0] if candidate_tables else None
    if table is None:
        raise ValueError("Aucun tableau trouvé sur la page")

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        texts = [td.get_text(" ", strip=True) for td in tds]
        if len(texts) < 3:
            continue
        # Heuristic mapping: try to detect common patterns
        # We'll try to map fields by typical Geny column order:
        # Num | Cheval | SA/Dist | Driver | Entraineur | Musique | Gains | Rapports...
        # But pages vary — so use content-based extraction.
        row = {
            "Nom": "",
            "Numéro de corde": "",
            "Cote": "",
            "Poids": "",
            "Musique": "",
            "Âge/Sexe": "",
            "Jockey": "",
            "Entraîneur": "",
            "Gains": ""
        }
        # try to find a token that looks like a cote (contains ',' or '.') or digits
        cote_candidate = None
        gains_candidate = None
        musique_candidate = None
        # search within texts for music pattern (digits with a/A etc.)
        for t in texts:
            if re.search(r"\d+[a-zA-Z]\d+", t.replace(" ", "")):
                musique_candidate = t
            if re.search(r"\d+(\,\d+)?\s*$", t):  # ending with number -> maybe gains/cote
                # decide by size: if > 50 it's likely gains (like 129 180) otherwise cote
                raw_digits = re.sub(r"[^\d]", "", t)
                try:
                    val = int(raw_digits) if raw_digits else 0
                    if val > 1000:  # gains
                        gains_candidate = t
                    else:
                        # could be cote with comma
                        if "," in t or "." in t:
                            cote_candidate = t
                except:
                    pass
        # fallback mapping by positions for typical table lengths
        L = len(texts)
        try:
            if L >= 8:
                row["Numéro de corde"] = texts[0]
                row["Nom"] = texts[1]
                # look for jockey/driver in columns
                # try to pick columns that look like names (contain letters and spaces)
                row["Musique"] = musique_candidate or texts[5] if L>5 else ""
                # Gains often near the end
                row["Gains"] = gains_candidate or texts[-2] if L>1 else ""
                # Cote often last col
                row["Cote"] = cote_candidate or texts[-1]
            else:
                # short table — assume [num, name, musique, cote]
                row["Numéro de corde"] = texts[0]
                row["Nom"] = texts[1]
                if L > 2:
                    row["Musique"] = musique_candidate or texts[2]
                if L > 3:
                    row["Cote"] = cote_candidate or texts[-1]
        except Exception:
            # best effort
            row["Nom"] = texts[0]
            if L>1:
                row["Cote"] = texts[-1]
        # cleanup
        row = {k: clean_text(v) for k,v in row.items()}
        # attempt to clean Cote and Gains numeric
        # Cote examples: "13,9" or "13.9" or "—" or ""
        if row.get("Cote"):
            c = row["Cote"].replace(",", ".").strip()
            m = re.search(r"\d+(\.\d+)?", c)
            if m:
                row["Cote"] = float(m.group(0))
            else:
                row["Cote"] = np.nan
        else:
            row["Cote"] = np.nan
        # Gains: remove non-digit
        if row.get("Gains"):
            g = re.sub(r"[^\d]", "", row["Gains"])
            try:
                row["Gains"] = int(g) if g else 0
            except:
                row["Gains"] = 0
        else:
            row["Gains"] = 0
        rows.append(row)

    if not rows:
        raise ValueError("Aucune ligne extraite du tableau")

    df = pd.DataFrame(rows)

    # ensure key columns exist
    if "Numéro de corde" not in df.columns:
        df["Numéro de corde"] = range(1, len(df)+1)
    # ensure name normalization
    df["Nom"] = df["Nom"].apply(lambda s: re.sub(r"[^\w\s'\-À-ÿ]", "", str(s)))
    # fill missing columns with defaults
    for col in ["Poids", "Âge/Sexe", "Jockey", "Entraîneur", "Musique"]:
        if col not in df.columns:
            df[col] = ""
    # fallback Cote numeric
    df["Cote"] = df["Cote"].apply(lambda x: np.nan if pd.isna(x) else safe_float(x, default=np.nan))
    df["Cote"] = df["Cote"].fillna(999)  # sentinel for missing cote
    # Poids -> try extract numbers if present
    df["Poids"] = df["Poids"].apply(lambda x: extract_weight(x) if x else 60.0)
    return df[["Nom","Numéro de corde","Cote","Poids","Musique","Âge/Sexe","Jockey","Entraîneur","Gains"]]

# ------------------- Feature engineering -------------------
def music_to_features(music):
    s = str(music)
    digits = [int(x) for x in re.findall(r"\d+", s)]
    if not digits:
        return 0, 0, 0.0
    recent_wins = sum(1 for d in digits if d==1)
    recent_top3 = sum(1 for d in digits if d<=3)
    weights = np.linspace(1.0, 0.3, num=len(digits))
    weighted = sum((4 - d) * w for d,w in zip(digits, weights)) / (len(digits)+1e-6)
    return recent_wins, recent_top3, weighted

def prepare_data(df):
    df = df.copy()
    # normalize expected columns
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

# ------------------- History management -------------------
def append_to_historique(raw_df):
    df_copy = raw_df.copy()
    df_copy["_source_ts"] = datetime.now().isoformat()
    if os.path.exists(HIST_PATH):
        try:
            old = pd.read_csv(HIST_PATH)
            combined = pd.concat([old, df_copy], ignore_index=True)
        except Exception:
            combined = df_copy
    else:
        combined = df_copy
    combined.to_csv(HIST_PATH, index=False)
    return len(combined)

# ------------------- Hybrid model manager -------------------
class HybridModel:
    def __init__(self, feature_cols=None):
        self.feature_cols = feature_cols or ["odds_numeric","draw_numeric","weight_kg","age","is_female","recent_wins","recent_top3","recent_weighted"]
        self.scaler = StandardScaler()
        self.dl_model = None
        self.xgb_model = None
        # load if exists
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
        # fit scaler
        Xs = self.scaler.fit_transform(X_np)
        # DL
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
        # XGBoost
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
                st.warning(f"Erreur entraînement XGBoost: {e}")
                self.xgb_model = None
        # persist scaler
        try:
            joblib.dump(self.scaler, SCALER_PATH)
        except Exception as e:
            st.warning(f"Erreur sauvegarde scaler: {e}")
        # log
        loss = None
        if hist is not None:
            loss = hist.history.get("loss", [None])[-1]
        with open(TRAIN_LOG, "a") as f:
            f.write(f"{datetime.now().isoformat()},{len(Xs)},{loss},{DL_MODEL_PATH}\n")
        return hist.history if hist is not None else None

    def predict(self, X_df):
        if len(X_df) == 0:
            return np.array([]), np.array([]), np.array([])
        X_np = X_df.values if hasattr(X_df, "values") else np.array(X_df)
        Xs = self.scaler.transform(X_np)
        dl_preds = np.zeros(len(Xs))
        xgb_preds = np.zeros(len(Xs))
        if self.dl_model is not None and tf is not None:
            try:
                dl_preds = self.dl_model.predict(Xs).flatten()
            except Exception:
                dl_preds = np.zeros(len(Xs))
        if self.xgb_model is not None:
            try:
                xgb_preds = self.xgb_model.predict(Xs)
            except Exception:
                xgb_preds = np.zeros(len(Xs))
        # normalize
        def norm(a):
            a = np.array(a, dtype=float)
            if a.max() != a.min():
                return (a - a.min())/(a.max()-a.min())
            return np.zeros_like(a)
        ndl = norm(dl_preds) if dl_preds.size else np.zeros(len(Xs))
        nx = norm(xgb_preds) if xgb_preds.size else np.zeros(len(Xs))
        ensemble = 0.6*ndl + 0.4*nx if (nx.size and ndl.size) else (ndl if ndl.size else nx)
        return ensemble, ndl, nx

# ------------------- Helpers: combos, confidence -------------------
def compute_confidence(a, b):
    if a.size==0:
        return np.zeros_like(b)
    if b.size==0:
        return np.ones_like(a)
    arr = np.vstack([a, b])
    std = np.std(arr, axis=0)
    conf = 1.0 - (std / (std.max()+1e-9))
    return np.clip(conf, 0.0, 1.0)

def generate_weighted_trios(df_ranked, n=35, max_total_odds=120.0, avoid_same_trainer=True):
    names = df_ranked["Nom"].tolist()
    if len(names) < 3:
        return []
    scores = df_ranked["score_final"].values
    scores = np.maximum(scores, 1e-9)
    probs = scores / scores.sum()
    combos_set = set(); results=[]
    attempts = 0; max_attempts = max(2000, n*200)
    while len(results) < n and attempts < max_attempts:
        attempts += 1
        chosen = list(np.random.choice(names, size=3, replace=False, p=probs))
        key = tuple(sorted(chosen))
        if key in combos_set: continue
        sub = df_ranked[df_ranked["Nom"].isin(chosen)]
        try:
            total_odds = sub["odds_numeric"].astype(float).sum()
            if total_odds > max_total_odds: continue
        except:
            pass
        if avoid_same_trainer and "Entraîneur" in df_ranked.columns:
            trainers = sub.get("Entraîneur", pd.Series([""]*len(sub))).astype(str).tolist()
            if len(set(trainers)) < len(trainers): continue
        combos_set.add(key); results.append(tuple(chosen))
    return results

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="🏇 Analyseur Hippique Final", layout="wide", page_icon="🏇")
st.title("🏇 Analyseur Hippique — Scraper Geny + Auto-Training Hybride")

with st.sidebar:
    st.header("Configuration")
    use_scraper = st.checkbox("Utiliser scraper Geny automatique (URL)", value=True)
    auto_train = st.checkbox("Auto Train (historique)", value=True)
    dl_epochs = st.number_input("DL epochs", min_value=2, max_value=400, value=16, step=2)
    dl_batch = st.number_input("DL batch", min_value=2, max_value=64, value=8, step=1)
    use_cv = st.checkbox("Use XGBoost CV if historique large", value=True)
    blend_weight = st.slider("Poids modèles (1=100% modèles, 0=heuristique)", 0.0, 1.0, 0.6)
    n_combos = st.number_input("Nb combinaisons e-trio", min_value=5, max_value=200, value=35, step=1)
    st.markdown("---")
    st.info("Le script fusionne les imports dans data/historique.csv et entraînera les modèles si 'Auto Train' est activé.")

tab1, tab2, tab3 = st.tabs(["URL / Scraper", "Upload CSV", "Historique & Tests"])

df_race = None
with tab1:
    st.subheader("Scraper URL")
    url = st.text_input("URL de la page course (Geny ou tableau HTML)")
    if st.button("Scraper / Charger"):
        if not url:
            st.error("Fourni une URL")
        else:
            try:
                df_race = scrape_geny_course(url)
                st.success(f"Extraction OK ({len(df_race)} lignes)")
                st.dataframe(df_race.head(30))
            except Exception as e:
                st.error(f"Erreur scraping: {e}")

with tab2:
    st.subheader("Upload CSV (nouvelle course)")
    uploaded = st.file_uploader("CSV (colonnes attendues: Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe, optional: Jockey, Entraîneur, Gains)", type=["csv"])
    if uploaded:
        try:
            df_race = pd.read_csv(uploaded)
            st.success(f"Fichier chargé ({len(df_race)} lignes)")
            st.dataframe(df_race.head())
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")

with tab3:
    st.subheader("Historique & Tests")
    if st.button("Afficher historique (dernieres 50 lignes)"):
        if os.path.exists(HIST_PATH):
            hist = pd.read_csv(HIST_PATH)
            st.write(f"Historique total: {len(hist)} lignes")
            st.dataframe(hist.tail(50))
        else:
            st.info("Aucun historique trouvé")
    if st.button("Supprimer historique"):
        if os.path.exists(HIST_PATH):
            os.remove(HIST_PATH); st.success("Historique supprimé")
        else:
            st.info("Aucun historique à supprimer")

# Main pipeline
if df_race is not None and len(df_race) > 0:
    st.markdown("---")
    st.header("Préparation & Entraînement")
    df_prep = prepare_data(df_race)
    st.dataframe(df_prep[["Nom","Cote","odds_numeric","draw_numeric","weight_kg","recent_wins","recent_top3","recent_weighted"]].assign(heuristic=lambda d: (1/(d["odds_numeric"]+0.1)).round(3)))

    # append to historique
    total = append_to_historique(df_race)
    st.success(f"Historique mis à jour — total lignes: {total}")

    # build model manager
    feats = ["odds_numeric","draw_numeric","weight_kg","age","is_female","recent_wins","recent_top3","recent_weighted"]
    manager = HybridModel(feature_cols=feats)

    # prepare X/y from historique (preferred) or pseudo-target
    X_hist = None; y_hist = None
    if os.path.exists(HIST_PATH):
        try:
            hist_raw = pd.read_csv(HIST_PATH)
            hist_prep = prepare_data(hist_raw)
            if "placement" in hist_raw.columns or "rank" in hist_raw.columns:
                if "placement" in hist_raw.columns:
                    y_hist = 1.0 / (hist_prep["placement"].astype(float) + 0.1)
                else:
                    y_hist = 1.0 / (hist_prep["rank"].astype(float) + 0.1)
                X_hist = hist_prep[feats]
            else:
                # pseudo target
                y_hist = 0.7*(1.0/(hist_prep["odds_numeric"]+0.1)) + 0.3*(hist_prep["recent_weighted"]/(hist_prep["recent_weighted"].max()+1e-6))
                X_hist = hist_prep[feats]
        except Exception as e:
            st.warning(f"Erreur lecture historique: {e}")

    trained = False
    if auto_train:
        try:
            if X_hist is not None and len(X_hist) >= 4:
                st.info(f"Entraînement automatique sur historique ({len(X_hist)} échantillons)...")
                manager.train(X_hist, y_hist, dl_epochs=dl_epochs, dl_batch=dl_batch, use_cv=use_cv)
                trained = True
            else:
                # fallback train on current race if possible
                if len(df_prep) >= 3:
                    y_curr = 0.7*(1.0/(df_prep["odds_numeric"]+0.1)) + 0.3*(df_prep["recent_weighted"]/(df_prep["recent_weighted"].max()+1e-6))
                    st.info("Entrainement sur course courante (pseudo-target)...")
                    manager.train(df_prep[feats], y_curr, dl_epochs=max(4,dl_epochs//2), dl_batch=dl_batch, use_cv=False)
                    trained = True
        except Exception as e:
            st.warning(f"Erreur auto-train: {e}")

    # predictions
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

    # Display outputs
    left, right = st.columns([2,1])
    with left:
        st.subheader("Classement final")
        disp = df_ranked[["rang","Nom","Cote","Numéro de corde","weight_kg","score_final","confidence"]].copy()
        disp["Score"] = disp["score_final"].round(3)
        disp["Conf"] = (disp["confidence"]*100).round(1).astype(str) + "%"
        disp = disp.drop(["score_final","confidence"], axis=1)
        st.dataframe(disp, use_container_width=True)
    with right:
        st.subheader("Infos")
        st.metric("Nb chevaux", len(df_ranked))
        st.markdown(f"- Auto-train: **{'ON' if auto_train else 'OFF'}**")
        st.markdown(f"- Modèles chargés: DL={'Oui' if (manager.dl_model is not None) else 'Non'}, XGB={'Oui' if (manager.xgb_model is not None) else 'Non'}")
        if trained:
            st.success("Auto-train effectué")

    st.subheader("Visualisations")
    st.bar_chart(df_ranked.set_index("Nom")["score_final"].head(12))

    st.subheader("Générateur e-trio pondéré & filtré")
    combos = generate_weighted_trios(df_ranked, n=int(n_combos), max_total_odds=120.0, avoid_same_trainer=True)
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
st.info("Conseils: pour de bonnes performances il faut un historique labellisé (placements). Je peux t'aider à automatiser la récupération des résultats et compléter l'historique.")
