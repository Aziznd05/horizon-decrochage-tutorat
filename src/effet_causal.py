import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from pathlib import Path

DATA_DIR = Path("data/processed")

def estimate_causal_effect():
    print("Chargement des donnees...")
    df = pd.read_csv(DATA_DIR / "base_survie.csv")
    
    # 1. ANALYSE LANDMARK (On se place a la semaine 8)
    landmark = 8
    print(f"Filtrage : on ne garde que ceux qui ont survecu jusqu'a la semaine {landmark}...")
    
    # On ne garde que ceux qui ont un T > 8 (qui n'ont pas encore decroche avant ou a la semaine 8)
    df_landmark = df[df["T"] > landmark].copy()
    
    # On recale le temps (T devient le temps APRES la semaine 8)
    df_landmark["T_apres"] = df_landmark["T"] - landmark
    
    # On cree la variable de "traitement" : a-t-il commence le tutorat avant ou a la semaine 8 ?
    df_landmark["tutore_a_8"] = np.where(df_landmark["semaine_debut_tutorat"] <= landmark, 1, 0)
    # Ceux qui n'ont pas de semaine_debut_tutorat sont des NaN, on met 0
    df_landmark["tutore_a_8"] = df_landmark["tutore_a_8"].fillna(0).astype(int)
    
    print(f"Taille de l'echantillon apres filtrage : {len(df_landmark)} etudiants")
    print(f"Nombre de tutores a la semaine 8 : {df_landmark['tutore_a_8'].sum()}")

    # 2. SCORE DE PROPENSION (Probabilite d'etre tutore)
    # On utilise les variables de base pour predire qui recoit le tutorat
    features = ["age", "note_bac", "boursier", "genre"]
    df_ps = df_landmark.dropna(subset=features + ["T_apres", "E"]).copy()
    
    # Transformer les textes en chiffres
    df_ps = pd.get_dummies(df_ps, columns=["boursier", "genre"], drop_first=True)
    X = df_ps[["age", "note_bac", "boursier_Oui", "genre_M"]]
    y = df_ps["tutore_a_8"]
    
    # Modele de regression logistique
    model_ps = LogisticRegression()
    model_ps.fit(X, y)
    
    # Calcul des probabilites (le score)
    df_ps["propensity"] = model_ps.predict_proba(X)[:, 1]
    
    # 3. PONDERATION IPTW (Inverse Probability of Treatment Weighting)
    print("\nCalcul des poids IPTW...")
    df_ps["weight"] = np.where(df_ps["tutore_a_8"] == 1, 
                               1 / df_ps["propensity"], 
                               1 / (1 - df_ps["propensity"]))
    
    # Verification de l'equilibre (les moyennes doivent etre proches)
    print("\n--- Verification de l'equilibre (Moyennes ponderees) ---")
    mean_treated = np.average(df_ps.loc[df_ps["tutore_a_8"] == 1, "age"], weights=df_ps.loc[df_ps["tutore_a_8"] == 1, "weight"])
    mean_control = np.average(df_ps.loc[df_ps["tutore_a_8"] == 0, "age"], weights=df_ps.loc[df_ps["tutore_a_8"] == 0, "weight"])
    print(f"Moyenne d'age - Tutores : {mean_treated:.2f} | Non-tutores : {mean_control:.2f}")

    # 4. MODELE DE COX PONDERE
    print("\n--- Estimation de l'effet causal du tutorat ---")
    # On cree un modele de Cox avec le poids
    cols_cox = ["T_apres", "E", "tutore_a_8", "weight"]
    cph = CoxPHFitter()
    cph.fit(df_ps[cols_cox], duration_col="T_apres", event_col="E", weights_col="weight")
    
    cph.print_summary()
    
    hr = cph.hazard_ratios_["tutore_a_8"]
    print(f"\nHazard Ratio (HR) pondere pour le tutorat : {hr:.3f}")
    if hr < 1:
        print(f"-> Le tutorat REDUIT le risque de decrochage de {(1-hr)*100:.1f}% (C'est un effet protecteur).")
    else:
        print(f"-> Le tutorat AUGMENTE le risque de decrochage de {(hr-1)*100:.1f}% (Attention, biais possible).")

if __name__ == "__main__":
    estimate_causal_effect()