import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/etudiants.csv")
PROCESSED_PATH = Path("data/processed/etudiants.csv")

def clean_etudiants():
    print("Chargement des données brutes...")
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False, na_values=[""])
    
    # --- CORRECTION ICI : On nettoie les noms de colonnes (enlève les espaces) ---
    df.columns = df.columns.str.strip()
    # ----------------------------------------------------------------------------

    print(f"Lignes initiales : {len(df)}")

    # 1. ID Étudiant : Nettoyage des espaces et majuscules
    df["id_etudiant"] = df["id_etudiant"].str.strip().str.upper()

    # 2. Genre : Harmonisation (M / F)
    df["genre"] = df["genre"].str.strip().str.upper()
    mapping_genre = {
        "M": "M", "MASCULIN": "M", "HOMME": "M", "H": "M",
        "F": "F", "FEMININ": "F", "FEMME": "F"
    }
    df["genre"] = df["genre"].map(mapping_genre).fillna(df["genre"])

    # 3. Âge : Nettoyage et bornes réalistes
    df["age"] = df["age"].str.replace("ans", "", regex=False).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 16) | (df["age"] > 60), "age"] = np.nan

    # 4. Boursier : Harmonisation (Oui / Non)
    df["boursier"] = df["boursier"].str.strip().str.upper()
    mapping_binaire = {
        "OUI": "Oui", "O": "Oui", "1": "Oui", "YES": "Oui", "Y": "Oui",
        "NON": "Non", "N": "Non", "0": "Non", "NO": "Non"
    }
    df["boursier"] = df["boursier"].map(mapping_binaire).fillna(df["boursier"])

    # 5. Filière : Harmonisation
    df["filiere"] = df["filiere"].str.strip().str.upper()
    mapping_filiere = {
        "INFO": "Informatique", "INFORMATIQUE": "Informatique",
        "GC": "Génie Civil", "GENIE CIVIL": "Génie Civil", "GÉNIE CIVIL": "Génie Civil",
        "GESTION": "Gestion",
        "COM": "Communication", "COMMUNICATION": "Communication",
        "DROIT": "Droit"
    }
    df["filiere"] = df["filiere"].map(mapping_filiere).fillna(df["filiere"].str.title())

    # 6. Mention Bac : Harmonisation
    df["mention_bac"] = df["mention_bac"].str.strip().str.upper()
    mapping_mention = {
        "P": "Passable", "PASSABLE": "Passable",
        "AB": "Assez Bien", "ASSEZ BIEN": "Assez Bien",
        "B": "Bien", "BIEN": "Bien",
        "TB": "Très Bien", "TRES BIEN": "Très Bien", "TRÈS BIEN": "Très Bien"
    }
    df["mention_bac"] = df["mention_bac"].map(mapping_mention).fillna(df["mention_bac"].str.title())

    # 7. Note Bac : Virgule -> Point, et bornes [0, 20]
    df["note_bac"] = df["note_bac"].str.replace(",", ".", regex=False)
    df["note_bac"] = pd.to_numeric(df["note_bac"], errors="coerce")
    df.loc[(df["note_bac"] < 0) | (df["note_bac"] > 20), "note_bac"] = np.nan

    # 8. Région : Harmonisation basique
    df["region_origine"] = df["region_origine"].str.strip()
    df["region_origine"] = df["region_origine"].str.replace("DAKAR", "Dakar", regex=False)
    df["region_origine"] = df["region_origine"].str.replace("THIES", "Thiès", regex=False)
    df["region_origine"] = df["region_origine"].str.replace("Thies", "Thiès", regex=False)
    df["region_origine"] = df["region_origine"].str.replace("Zig", "Ziguinchor", regex=False)
    df["region_origine"] = df["region_origine"].str.replace("Saint Louis", "Saint-Louis", regex=False)
    df["region_origine"] = df["region_origine"].str.title()

    # 9. Distance : Nettoyage des "km", virgules, et valeurs aberrantes
    # On renomme la colonne pour enlever les espaces
    df = df.rename(columns={"Distance campus (km)": "distance_campus_km"})
    df["distance_campus_km"] = df["distance_campus_km"].str.replace("km", "", regex=False).str.replace(",", ".", regex=False).str.strip()
    df["distance_campus_km"] = pd.to_numeric(df["distance_campus_km"], errors="coerce")
    df.loc[(df["distance_campus_km"] < 0) | (df["distance_campus_km"] > 500), "distance_campus_km"] = np.nan

    # 10. Emploi parallèle : Oui / Non
    df["emploi_parallele"] = df["emploi_parallele"].str.strip().str.upper()
    df["emploi_parallele"] = df["emploi_parallele"].map(mapping_binaire).fillna(df["emploi_parallele"])

    # 11. Suppression des doublons sur l'ID
    avant = len(df)
    df = df.drop_duplicates(subset=["id_etudiant"], keep="first")
    if avant > len(df):
        print(f"Doublons supprimés : {avant - len(df)}")

    # Sauvegarde
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Fichier nettoyé sauvegardé dans : {PROCESSED_PATH}")

if __name__ == "__main__":
    clean_etudiants()