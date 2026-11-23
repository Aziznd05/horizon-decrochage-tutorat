import pandas as pd
import numpy as np
import re
from pathlib import Path

RAW_PATH = Path("data/raw/messages.csv")
PROCESSED_PATH = Path("data/processed/messages.csv")

def clean_messages():
    print("Chargement des données brutes...")
    df = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False, na_values=[""])
    print(f"Lignes initiales : {len(df)}")

    # 1. Nettoyage basique
    df["id_message"] = df["id_message"].str.strip().str.upper()
    df["id_etudiant"] = df["id_etudiant"].str.strip().str.upper()
    df["canal"] = df["canal"].str.strip().str.lower()
    df["semaine"] = pd.to_numeric(df["semaine"], errors="coerce")
    df.loc[(df["semaine"] < 1) | (df["semaine"] > 30), "semaine"] = np.nan

    # 2. Correction de l'encodage (Mojibake)
    remplacements_encodage = {
        "Ã©": "é", "Ã¨": "è", "Ã ": "à", "Ã§": "ç", "Ãª": "ê", "Ã¹": "ù"
    }
    for mauvais, bon in remplacements_encodage.items():
        df["texte"] = df["texte"].str.replace(mauvais, bon, regex=False)

    # 3. Pseudonymisation des données personnelles (PII)
    # On crée une colonne 'texte_pseudo'
    df["texte_pseudo"] = df["texte"]
    
    # a) Masquer les emails
    df["texte_pseudo"] = df["texte_pseudo"].str.replace(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
        '[EMAIL_MASQUE]', 
        regex=True
    )
    
    # b) Masquer les numéros de téléphone (format Sénégalais : 77, 78, 76, 70, 75 + 7 chiffres)
    df["texte_pseudo"] = df["texte_pseudo"].str.replace(
        r'\b(?:77|78|76|70|75)\s?\d{3}\s?\d{2}\s?\d{2}\b', 
        '[TEL_MASQUE]', 
        regex=True
    )
    
    # c) Masquer les noms/prénoms (ex: "je suis Moussa Gueye", "c'est Awa Ndiaye")
    # On cible les phrases type "Je suis [Prénom] [Nom]" ou "c'est [Prénom] [Nom]"
    df["texte_pseudo"] = df["texte_pseudo"].str.replace(
        r'(?i)(c\'est|je suis|je m\'appelle)\s+[A-Z][a-zA-ZÀ-ÿ-]+\s+[A-Z][a-zA-ZÀ-ÿ-]+', 
        r'\1 [NOM_MASQUE]', 
        regex=True
    )

    # 4. Suppression des messages vides ou inutiles
    df["texte_pseudo"] = df["texte_pseudo"].str.strip()
    df = df[~df["texte_pseudo"].isin(["", ".", ",", " "])]
    df = df.dropna(subset=["texte_pseudo"])

    # 5. Suppression des doublons éventuels
    avant = len(df)
    df = df.drop_duplicates(subset=["id_message"], keep="first")
    if avant > len(df):
        print(f"Doublons supprimés : {avant - len(df)}")

    # Sauvegarde
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Fichier nettoyé et pseudonymisé sauvegardé dans : {PROCESSED_PATH}")
    print(f"Lignes conservées : {len(df)}")

if __name__ == "__main__":
    clean_messages()