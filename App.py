# -*- coding: utf-8 -*-
"""
🏇 Analyseur Hippique IA — Auto-Entraînement & Historique
Scraping Geny.com + Deep Learning + apprentissage continu
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests, os, re, joblib, warnings
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks

warnings.filterwarnings('ignore')
st.set_page_config(page_title="🏇 Analyseur Hippique IA", page_icon="🐎", layout="wide")

# ---------------- CONFIG ----------------
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
HISTORIQUE_PATH = os.path.join(DATA_DIR, "historique_courses.csv")

# ---------------- UTILS ----------------
def safe_float(val, default=0.0):
    try:
        return float(str(val).replace(',', '.'))
    except:
        return default

def music_to_score(music):
    digits = [int(x) for x in re.findall(r'\d+', str(music))]
    if not digits:
        return 0, 0, 0
    recent_win = sum(d == 1 for d in digits)
    top3 = sum(d <= 3 for d in digits)
    weights = np.linspace(1, 0.3, num=len(digits))
    weighted = np.sum((4 - np.array(digits)) * weights) / (len(digits) + 1e-6)
    return recent_win, top3, weighted

# ---------------- SCRAPER ----------------
@st.cache_data(ttl=600)
def scrape_geny_course(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, f"Erreur HTTP {r.status_code}"

        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return None, "Aucun tableau détecté"

        rows = table.find_all("tr")[1:]
        data = []
        for row in rows:
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) >= 5:
                data.append({
                    "Numéro": cols[0],
                    "Cheval": cols[1],
                    "Musique": cols[2],
                    "Driver/Jockey": cols[3],
                    "Entraîneur": cols[4] if len(cols) > 4 else "",
                    "Cote": cols[-1] if len(cols) >= 6 else "0",
                    "Gains": cols[-2] if len(cols) >= 6 else "0"
                })
        if not data:
            return None, "Aucune donnée extraite"
        df = pd.DataFrame(data)
        return df, "Succès"
    except Exception as e:
        return None, f"Erreur scraping: {e}"

# ---------------- MODEL DL ----------------
class ModelDL:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, "model_geny.keras")
        self.scaler_path = os.path.join(MODEL_DIR, "scaler_geny.joblib")

    def build(self, input_dim):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, X, y, epochs=60):
        Xs = self.scaler.fit_transform(X)
        model = self.build(Xs.shape[1])
        cb = [callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
        hist = model.fit(Xs, y, epochs=epochs, validation_split=0.2, verbose=0, callbacks=cb)
        self.model, self.history = model, hist.history
        model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def predict(self, X):
        if self.model is None:
            if os.path.exists(self.model_path):
                self.model = models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
            else:
                return np.zeros(len(X))
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs).flatten()

# ---------------- ANALYSE ----------------
def prepare_data(df):
    df = df.copy()
    df["Cote_num"] = df["Cote"].apply(lambda x: safe_float(x, 99))
    df["Gains_num"] = df["Gains"].apply(lambda x: safe_float(re.sub(r"[^\d,\.]", "", x), 0))
    df["recent_win"], df["top3"], df["weighted"] = zip(*df["Musique"].apply(music_to_score))
    df = df.fillna(0)
    return df

def generate_predictions(df, model_dl):
    X = df[["Cote_num", "Gains_num", "recent_win", "top3", "weighted"]].values
    y_pseudo = 0.6 * (1 / (df["Cote_num"] + 0.1)) + 0.4 * (df["weighted"] / (df["weighted"].max() + 1e-6))
    model_dl.train(X, y_pseudo)
    preds = model_dl.predict(X)
    preds = (preds - preds.min()) / (preds.max() - preds.min() + 1e-6)
    df["Score"] = preds
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rang"] = df.index + 1
    return df

def update_historique(df, url):
    df_histo = pd.read_csv(HISTORIQUE_PATH) if os.path.exists(HISTORIQUE_PATH) else pd.DataFrame()
    df["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    df["URL"] = url
    df_histo = pd.concat([df_histo, df], ignore_index=True)
    df_histo.to_csv(HISTORIQUE_PATH, index=False)

# ---------------- UI ----------------
def main():
    st.markdown("<h1 style='text-align:center;color:#1e3a8a;'>🏇 Analyseur Hippique IA — Auto-Entraînement</h1>", unsafe_allow_html=True)
    st.write("Collez une URL de course **Geny.com** pour lancer l’analyse et améliorer le modèle à chaque utilisation.")

    url = st.text_input("🔍 URL de la course :", placeholder="https://www.geny.com/stats-pmu?id_course=...")
    lancer = st.button("Analyser la course")

    if lancer and url:
        with st.spinner("⏳ Extraction en cours..."):
            df, msg = scrape_geny_course(url)
            if df is None:
                st.error(f"❌ {msg}")
                return
            st.success(f"✅ {len(df)} chevaux extraits ({msg})")
            st.dataframe(df)

        with st.spinner("🧠 Entraînement et pronostic IA..."):
            df_prep = prepare_data(df)
            model = ModelDL()
            df_pred = generate_predictions(df_prep, model)
            update_historique(df_pred, url)

        st.header("🏆 Pronostic IA")
        st.dataframe(df_pred[["Rang", "Cheval", "Cote", "Gains", "Score"]].style.format({"Score": "{:.3f}"}))

        st.subheader("🥇 Top 3 Chevaux Prédits")
        for i in range(3):
            row = df_pred.iloc[i]
            st.markdown(f"<div style='border-left:4px solid #f59e0b;padding:8px;margin:4px;background:#fff8e1;'>"
                        f"<b>{i+1}. {row['Cheval']}</b> — Cote {row['Cote']} | Score {row['Score']:.3f}</div>", unsafe_allow_html=True)

        # Téléchargement
        st.download_button("📄 Télécharger le pronostic (CSV)", df_pred.to_csv(index=False), file_name="pronostic.csv", mime="text/csv")

        # Historique affiché
        if os.path.exists(HISTORIQUE_PATH):
            st.markdown("---")
            st.subheader("📚 Historique d'apprentissage (auto-train)")
            df_histo = pd.read_csv(HISTORIQUE_PATH)
            st.dataframe(df_histo.tail(10))

if __name__ == "__main__":
    main()
