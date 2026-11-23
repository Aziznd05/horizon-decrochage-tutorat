import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from pathlib import Path

DATA_DIR = Path("data/processed")
FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def run_cox_model():
    print("Chargement des donnees...")
    df = pd.read_csv(DATA_DIR / "base_survie.csv")
    
    # On selectionne les colonnes utiles
    cols = ["T", "E", "genre", "age", "boursier", "filiere", "note_bac", "emploi_parallele"]
    df_model = df[cols].copy()
    
    # On supprime les lignes avec des valeurs manquantes
    df_model = df_model.dropna()
    
    # On transforme les textes en 0 et 1
    df_model = pd.get_dummies(df_model, columns=["genre", "boursier", "filiere", "emploi_parallele"], drop_first=True)
    
    print(f"Nombre d'etudiants analyses : {len(df_model)}")

    # Creation et entrainement du modele
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col="T", event_col="E")
    
    # 1. Affichage des resultats dans le terminal
    print("\n--- Resultats du modele de Cox ---")
    cph.print_summary()
    
    # 2. NOUVEAU : Sauvegarde des resultats en fichier CSV
    # C'est beaucoup plus lisible que le format texte
    cph.summary.to_csv("docs/resultats_cox.csv")
    print("\nLes resultats ont ete sauvegardes dans docs/resultats_cox.csv")

    # 3. Verification des hypotheses (Schoenfeld)
    print("\n--- Verification des hypotheses ---")
    try:
        cph.check_assumptions(df_model, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print("Note : Certaines variables peuvent violer l'hypothese de proportionnalite.")

if __name__ == "__main__":
    run_cox_model()