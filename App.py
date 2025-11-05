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

    # Extraction des patterns de musique (nombre de courses récentes / victoires)
    data["musique_len"] = data["musique"].astype(str).apply(lambda x: len(re.findall(r"[0-9]+a", x)))
    data["victoires_recente"] = data["musique"].astype(str).apply(lambda x: x.count("1a"))

    # Nettoyage des variables numériques
    data["age"] = pd.to_numeric(data.get("age", 0), errors="coerce")
    data["poids"] = pd.to_numeric(data.get("poids", 0), errors="coerce")

    # 🔧 Nettoyage et extraction du rang d’arrivée
    if "arrivee" in data.columns:
        # Garder uniquement les chiffres, ignorer Da / NP / etc.
        data["arrivee_num"] = (
            data["arrivee"]
            .astype(str)
            .str.extract(r"(\d+)")
            .astype(float)
        )

        # Déterminer Top 3 uniquement si on a un rang 1 à 3
        data["top3"] = data["arrivee_num"].apply(lambda x: 1 if pd.notna(x) and 1 <= x <= 3 else 0)
    else:
        data["arrivee_num"] = 0
        data["top3"] = 0

    data = data.fillna(0)
    return data
