import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def build_survie(n_semaines_inactives=4):
    print("Chargement des donnees nettoyees...")
    etudiants = pd.read_csv(PROCESSED_DIR / "etudiants.csv")
    inscriptions = pd.read_csv(PROCESSED_DIR / "inscriptions.csv", sep=";")
    activite = pd.read_csv(PROCESSED_DIR / "activite_hebdo.csv")
    tutorat = pd.read_csv(PROCESSED_DIR / "tutorat.csv")

    # --- 1. TRAITEMENT DES ABANDONS EXPLICITES (Fichier inscriptions) ---
    inscriptions["date_debut"] = pd.to_datetime(inscriptions["date_debut"], errors="coerce")
    inscriptions["date_fin"] = pd.to_datetime(inscriptions["date_fin"], errors="coerce")
    
    # Calcul de la semaine de fin (difference de jours // 7 + 1)
    inscriptions["T_explicit"] = (inscriptions["date_fin"] - inscriptions["date_debut"]).dt.days // 7 + 1
    inscriptions["E_explicit"] = 0
    
    # Si abandon declare -> E=1
    inscriptions.loc[inscriptions["statut_final"] == "abandon", "E_explicit"] = 1
    # Si toujours inscrit (ou pas de date de fin) -> T=30, E=0
    inscriptions.loc[inscriptions["statut_final"] == "inscrit", "T_explicit"] = 30
    inscriptions["T_explicit"] = inscriptions["T_explicit"].fillna(30)
    # On plafonne a 30 semaines maximum
    inscriptions["T_explicit"] = inscriptions["T_explicit"].clip(upper=30)

    # --- 2. TRAITEMENT DES ABANDONS IMPLICITES (Fichier activite) ---
    print(f"Recherche des abandons implicites ({n_semaines_inactives} semaines consecutives)...")
    
    # Creation des tables pivot (etudiants en lignes, semaines en colonnes)
    act_conn = activite.pivot(index="id_etudiant", columns="semaine", values="nb_connexions_lms")
    act_dev = activite.pivot(index="id_etudiant", columns="semaine", values="devoirs_rendus")
    
    # On s'assure d'avoir bien les 30 semaines
    act_conn = act_conn.reindex(columns=range(1, 31))
    act_dev = act_dev.reindex(columns=range(1, 31))
    
    # Inactif = 0 connexion ET 0 devoir (on remplace les NaN par 0)
    inactive = (act_conn.fillna(0) == 0) & (act_dev.fillna(0) == 0)
    
    # Fonction pour trouver la premiere semaine d'une sequence de N semaines inactives
    def get_implicit_week(row):
        consecutive = 0
        for week, is_inactive in row.items():
            if is_inactive:
                consecutive += 1
                if consecutive == n_semaines_inactives:
                    return week - n_semaines_inactives + 1
            else:
                consecutive = 0
        return np.nan

    # Application de la fonction sur chaque etudiant
    T_implicit = inactive.apply(get_implicit_week, axis=1).rename("T_implicit")

    # --- 3. FUSION ET LOGIQUE FINALE ---
    df = etudiants[["id_etudiant"]].copy()
    df = df.merge(inscriptions[["id_etudiant", "T_explicit", "E_explicit"]], on="id_etudiant", how="left")
    df = df.merge(T_implicit, left_on="id_etudiant", right_index=True, how="left")

    df["T"] = 30
    df["E"] = 0

    # Appliquer les abandons/transferts explicites
    df.loc[df["E_explicit"] == 1, "E"] = 1
    df.loc[df["E_explicit"] == 1, "T"] = df["T_explicit"]
    df.loc[df["T_explicit"] < 30, "T"] = df["T_explicit"] # Transferts (E=0)

    # Appliquer les abandons implicites (s'ils surviennent AVANT le T actuel)
    mask_implicit = df["T_implicit"].notna() & (df["T_implicit"] < df["T"])
    df.loc[mask_implicit, "T"] = df.loc[mask_implicit, "T_implicit"]
    df.loc[mask_implicit, "E"] = 1

    # --- 4. AJOUT DES COVARIABLES ET DU TUTORAT ---
    # Infos etudiants
    df = df.merge(etudiants, on="id_etudiant", how="left")
    
    # Infos tutorat
    tuto_clean = tutorat.groupby("id_etudiant").agg(
        tutorat=("id_etudiant", "size"),
        semaine_debut_tutorat=("semaine_debut", "min")
    ).reset_index()
    
    df = df.merge(tuto_clean, on="id_etudiant", how="left")
    df["tutorat"] = df["tutorat"].fillna(0).astype(int)

    # --- 5. SAUVEGARDE ---
    cols = ["id_etudiant", "T", "E", "tutorat", "semaine_debut_tutorat", 
            "cohorte", "genre", "age", "boursier", "filiere", "mention_bac", 
            "note_bac", "region_origine", "distance_campus_km", "emploi_parallele"]
    
    base_survie = df[cols]
    base_survie.to_csv(PROCESSED_DIR / "base_survie.csv", index=False)
    print(f"base_survie.csv sauvegardee ({len(base_survie)} etudiants).")

    # --- 6. FORMAT PERIODE-PERSONNE ---
    print("Creation de base_periode.csv (format periode-personne)...")
    records = []
    for row in base_survie.itertuples():
        for w in range(1, int(row.T) + 1):
            is_event = 1 if (w == row.T and row.E == 1) else 0
            tuto_en_cours = 1 if (row.tutorat == 1 and row.semaine_debut_tutorat <= w) else 0
            records.append({
                "id_etudiant": row.id_etudiant,
                "semaine": w,
                "E": is_event,
                "tutorat": row.tutorat,
                "tutorat_en_cours": tuto_en_cours,
                "boursier": row.boursier,
                "genre": row.genre
            })
    df_periode = pd.DataFrame(records)
    df_periode.to_csv(PROCESSED_DIR / "base_periode.csv", index=False)
    print(f"base_periode.csv sauvegardee ({len(df_periode)} lignes).")

if __name__ == "__main__":
    build_survie(n_semaines_inactives=4)