import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/inscriptions.csv")
PROCESSED_PATH = Path("data/processed/inscriptions.csv")

def parse_dates(series):
    """Fonction pour gérer les formats de dates hétérogènes."""
    # Remplacement des mois en français par des chiffres
    mois_fr = {
        "janv.": "01", "févr.": "02", "mars": "03", "avr.": "04",
        "mai": "05", "juin": "06", "juil.": "07", "août": "08",
        "sept.": "09", "oct.": "10", "nov.": "11", "déc.": "12"
    }
    s = series.copy().astype(str)
    for mot, num in mois_fr.items():
        s = s.str.replace(mot, num, regex=False)
    
    # Conversion en datetime (dayfirst=True gère le format 06/10/2025)
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def clean_inscriptions():
    print("Chargement des données brutes...")
    # ATTENTION : sep=";" car le fichier utilise des points-virgules
    df = pd.read_csv(RAW_PATH, sep=";", dtype=str, keep_default_na=False, na_values=[""])
    
    # Nettoyage des noms de colonnes
    df.columns = df.columns.str.strip()
    print(f"Lignes initiales : {len(df)}")

    # 1. Identifiant
    df["id_etudiant"] = df["id_etudiant"].str.strip().str.upper()

    # 2. Statut final : Harmonisation
    df["statut_final"] = df["statut_final"].str.strip().str.lower()
    # Remplacement des accents et variations
    df["statut_final"] = df["statut_final"].str.replace("é", "e")
    mapping_statut = {
        "inscrit": "inscrit",
        "abandon": "abandon",
        "abandonne": "abandon",
        "transfert": "transfert",
        "transfere": "transfert"
    }
    df["statut_final"] = df["statut_final"].map(mapping_statut).fillna(df["statut_final"])

    # 3. Dates : Nettoyage
    df["date_debut"] = parse_dates(df["date_debut"])
    df["date_fin"] = parse_dates(df["date_fin"])

    # 4. Cohérence temporelle
    # Si date_fin < date_debut, on met date_fin à NaT (erreur de saisie)
    incoherence = df["date_fin"] < df["date_debut"]
    df.loc[incoherence, "date_fin"] = pd.NaT
    
    # 5. Suppression des doublons
    avant = len(df)
    df = df.drop_duplicates(subset=["id_etudiant"], keep="first")
    if avant > len(df):
        print(f"Doublons supprimés : {avant - len(df)}")

    # 6. Sauvegarde
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, sep=";") # On garde le ; pour la sortie
    print(f"Fichier nettoyé sauvegardé dans : {PROCESSED_PATH}")

if __name__ == "__main__":
    clean_inscriptions()