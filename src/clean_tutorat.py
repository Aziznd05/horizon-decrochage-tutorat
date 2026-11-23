import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/tutorat.csv")
PROCESSED_PATH = Path("data/processed/tutorat.csv")

def clean_tutorat():
    print("Chargement des données brutes...")
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False, na_values=[""])
    
    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"ID Etudiant": "id_etudiant"})
    
    print(f"Lignes initiales : {len(df)}")

    # 1. Identifiant
    df["id_etudiant"] = df["id_etudiant"].str.strip().str.upper()

    # 2. Semaine de début : doit être entre 1 et 30
    df["semaine_debut"] = pd.to_numeric(df["semaine_debut"], errors="coerce")
    df.loc[(df["semaine_debut"] < 1) | (df["semaine_debut"] > 30), "semaine_debut"] = np.nan

    # 3. Nombre de séances : doit être strictement positif
    df["nb_seances"] = pd.to_numeric(df["nb_seances"], errors="coerce")
    df.loc[df["nb_seances"] < 0, "nb_seances"] = np.nan

    # 4. Type de tuteur : Harmonisation
    df["type_tuteur"] = df["type_tuteur"].str.strip().str.lower()
    mapping_tuteur = {
        "pair": "Pair",
        "pair (étudiant avancé)": "Pair",
        "enseignant": "Enseignant",
        "prof": "Enseignant",
        "étudiant": "Étudiant"
    }
    df["type_tuteur"] = df["type_tuteur"].map(mapping_tuteur).fillna(df["type_tuteur"].str.title())

    # 5. Motif d'orientation : Harmonisation
    df["motif_orientation"] = df["motif_orientation"].str.strip().str.lower()

    # 6. Suppression des doublons stricts
    avant = len(df)
    df = df.drop_duplicates(subset=["id_etudiant", "semaine_debut"], keep="first")
    if avant > len(df):
        print(f"Doublons supprimés : {avant - len(df)}")

    # Sauvegarde
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Fichier nettoyé sauvegardé dans : {PROCESSED_PATH}")

if __name__ == "__main__":
    clean_tutorat()