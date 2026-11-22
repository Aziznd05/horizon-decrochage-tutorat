import pandas as pd
import numpy as np
from pathlib import Path

# Définition des chemins
RAW_PATH = Path("data/raw/activite_hebdo.csv")
PROCESSED_PATH = Path("data/processed/activite_hebdo.csv")

def clean_activite():
    print("Chargement des données brutes...")
    # Lecture en texte pour gérer les virgules et les valeurs vides
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False, na_values=[""])
    
    print(f"Lignes initiales : {len(df)}")
    
    # 1. Harmonisation des identifiants (majuscules)
    df["id_etudiant"] = df["id_etudiant"].str.upper()
    
    # 2. Conversion des notes (virgule -> point)
    df["note_moyenne"] = df["note_moyenne"].str.replace(",", ".", regex=False)
    
    # 3. Conversion des types (nombres)
    cols_numeric = ["semaine", "nb_connexions_lms", "devoirs_rendus", "note_moyenne", "absences"]
    for col in cols_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    # 4. Traitement des aberrations (on les transforme en NaN)
    df.loc[df["nb_connexions_lms"] < 0, "nb_connexions_lms"] = np.nan
    df.loc[df["devoirs_rendus"] > 2, "devoirs_rendus"] = np.nan
    df.loc[(df["note_moyenne"] < 0) | (df["note_moyenne"] > 20), "note_moyenne"] = np.nan
    
    # 5. Traitement de la sentinelle absences == 20 (code caché pour décrochage)
    df.loc[df["absences"] == 20, "absences"] = np.nan
    
    # 6. Suppression des doublons (même étudiant, même semaine)
    avant_doublons = len(df)
    df = df.drop_duplicates(subset=["id_etudiant", "semaine"], keep="first")
    print(f"Doublons supprimés : {avant_doublons - len(df)}")
    
    # 7. Tri propre par étudiant et par semaine
    df = df.sort_values(by=["id_etudiant", "semaine"]).reset_index(drop=True)
    
    # Sauvegarde du fichier propre
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Fichier nettoyé sauvegardé dans : {PROCESSED_PATH}")

if __name__ == "__main__":
    clean_activite()