import pandas as pd
import numpy as np
from pathlib import Path
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score

DATA_DIR = Path("data/processed")

def run_ml_survival():
    print("Chargement des donnees...")
    df = pd.read_csv(DATA_DIR / "base_survie.csv")
    
    # 1. Preparation des donnees
    cols = ["T", "E", "cohorte", "age", "note_bac", "boursier", "genre"]
    df_model = df[cols].dropna().copy()
    
    # On transforme les textes en chiffres (One-Hot Encoding)
    df_model = pd.get_dummies(df_model, columns=["boursier", "genre"], drop_first=True)
    
    # 2. Validation temporelle (Train 2024 / Test 2025)
    print("Separation des cohortes (Train: 2024, Test: 2025)...")
    df_train = df_model[df_model["cohorte"] == 2024]
    df_test = df_model[df_model["cohorte"] == 2025]
    
    # On separe les variables predictives (X) et la cible (y)
    features = [c for c in df_model.columns if c not in ["T", "E", "cohorte"]]
    X_train = df_train[features]
    X_test = df_test[features]
    
    # Format special pour scikit-survival
    y_train = Surv.from_arrays(event=df_train["E"].astype(bool), time=df_train["T"])
    y_test = Surv.from_arrays(event=df_test["E"].astype(bool), time=df_test["T"])
    
    # 3. Entrainement du modele Random Survival Forest
    print("Entrainement du modele Random Survival Forest...")
    rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, random_state=42, n_jobs=-1)
    rsf.fit(X_train, y_train)
    
    # 4. Evaluation sur la cohorte 2025
    print("\n--- Evaluation sur la cohorte 2025 ---")
    # Predire le risque (plus le chiffre est eleve, plus le risque est grand)
    risk_scores = rsf.predict(X_test)
    
    # Calcul du C-index (capacite a bien classer les etudiants)
    c_index = concordance_index_censored(df_test["E"].astype(bool), df_test["T"], risk_scores)[0]
    print(f"C-index (Random Survival Forest) : {c_index:.3f}")
    
    # Pour comparer, on calcule aussi le C-index d'un modele au hasard (0.5)
    print("Rappel : 0.5 = hasard, 1.0 = prediction parfaite.")
    if c_index > 0.6:
        print("-> Le modele ML est meilleur que le hasard, il a appris des choses utiles !")

    # 5. Sauvegarde des resultats pour le rapport
    with open("docs/resultats_ml.txt", "w") as f:
        f.write(f"C-index Random Survival Forest (Train 2024, Test 2025) : {c_index:.3f}\n")
        f.write("Modele entraine avec succes.\n")
    print("\nResultats sauvegardes dans docs/resultats_ml.txt")

if __name__ == "__main__":
    run_ml_survival()