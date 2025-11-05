import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =====================================================
# 1. Chargement automatique du fichier
# =====================================================
def load_course_file(file):
    try:
        df = pd.read_csv(file, sep=';', low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(file, sep=',', low_memory=False)
        except Exception:
            df = pd.read_excel(file)
    return df

# =====================================================
# 2. Mapping automatique des colonnes utiles
# =====================================================
def map_columns(df):
    mapping = {
        "cheval": ["cheval", "nom", "horse", "name"],
        "cote": ["cotedirect", "coteprob", "cote", "odds"],
        "jockey": ["jockey", "driver", "pilote"],
        "entraineur": ["entraineur", "trainer"],
        "musique": ["musiqueche", "musique", "music"],
        "age": ["age", "âge"],
        "poids": ["poidmont", "poids", "weight"],
        "arrivee": ["arrive", "rang", "arrivée", "place"]
    }
    cols_found = {}
    for key, patterns in mapping.items():
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    cols_found[key] = col
                    break
            if key in cols_found:
                break
    return cols_found

# =====================================================
# 3. Nettoyage et features
# =====================================================
def preprocess_data(df, cols_found):
    data = pd.DataFrame()

    for key, col in cols_found.items():
        data[key] = df[col]

    # Nettoyage des cotes
    data["cote"] = (
        data["cote"]
        .astype(str)
        .str.replace(",", ".")
        .str.extract(r"([\d.]+)")
        .astype(float)
    )

    # Longueur de la musique et nb de 1a = victoires
    data["musique_len"] = data["musique"].astype(str).apply(lambda x: len(re.findall(r"[0-9]+a", x)))
    data["victoires_recente"] = data["musique"].astype(str).apply(lambda x: x.count("1a"))

    # Âge, poids
    data["age"] = pd.to_numeric(data["age"], errors="coerce")
    data["poids"] = pd.to_numeric(data["poids"], errors="coerce")

    # Top 3 si présent
    if "arrivee" in data.columns:
        data["arrivee_num"] = pd.to_numeric(data["arrivee"], errors="coerce")
        data["top3"] = data["arrivee_num"].apply(lambda x: 1 if pd.notna(x) and x <= 3 else 0)
    else:
        data["top3"] = np.nan

    data = data.fillna(0)
    return data

# =====================================================
# 4. Entraînement du modèle ML
# =====================================================
def train_model(df):
    features = ["cote", "musique_len", "victoires_recente", "age", "poids"]
    X = df[features]
    y = df["top3"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# =====================================================
# 5. Interface Streamlit
# =====================================================
def app():
    st.set_page_config(page_title="Prédicteur PMU Geny", layout="wide")
    st.title("🐎 Prédicteur de classement Top 3 - PMU / Geny.fr")
    st.markdown("Ce modèle apprend sur vos historiques de courses Geny et estime la probabilité d’un **Top 3** pour chaque cheval.")

    uploaded_file = st.file_uploader("📂 Importez un fichier CSV/Excel (export Geny.fr)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df_raw = load_course_file(uploaded_file)
        st.success(f"✅ Fichier chargé : {df_raw.shape[0]} chevaux / {df_raw.shape[1]} colonnes")

        cols_found = map_columns(df_raw)
        st.write("🧭 Colonnes détectées :", cols_found)

        df = preprocess_data(df_raw, cols_found)

        # Si on a des arrivées (historique)
        if df["top3"].sum() > 0:
            model, acc = train_model(df)
            st.success(f"🎯 Modèle entraîné avec une précision de {acc*100:.2f}% sur l’échantillon de test.")

            # Prédiction sur l'ensemble
            features = ["cote", "musique_len", "victoires_recente", "age", "poids"]
            df["proba_top3"] = model.predict_proba(df[features])[:, 1]

            # Classement par probabilité décroissante
            results = df[["cheval", "cote", "musique", "jockey", "entraineur", "proba_top3"]]
            results = results.sort_values("proba_top3", ascending=False)
            results["proba_top3"] = (results["proba_top3"] * 100).round(1)

            st.subheader("🏆 Classement prédit (Top 3 probability)")
            st.dataframe(results.reset_index(drop=True).head(10))

            st.download_button(
                "📥 Télécharger les prédictions complètes",
                results.to_csv(index=False).encode("utf-8"),
                "predictions_top3.csv",
                "text/csv"
            )
        else:
            st.warning("⚠️ Ce fichier ne contient pas de résultats (‘arrivee’) — impossible d’entraîner un modèle. Ajoutez un historique de courses terminées.")

if __name__ == "__main__":
    app()
