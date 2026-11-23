import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from pathlib import Path

# Dossiers pour lire les données et sauvegarder les images
DATA_DIR = Path("data/processed")
FIG_DIR = Path("docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

def analyze_survival():
    print("Lecture du tableau des etudiants...")
    df = pd.read_csv(DATA_DIR / "base_survie.csv")
    
    # On enleve les lignes ou on ne connait pas le temps ou l'evenement
    df = df.dropna(subset=["T", "E"])

    # --- 1. LA COURBE GLOBALE ---
    print("\nCreation de la courbe globale...")
    kmf = KaplanMeierFitter()
    kmf.fit(df["T"], event_observed=df["E"], label="Tous les etudiants")
    
    # Affichage des chiffres cles dans le terminal
    print(f"Temps median avant decrochage : {kmf.median_survival_time_} semaines")
    print(f"Taux de survie a 10 semaines : {kmf.predict(10):.1%}")
    print(f"Taux de survie a 20 semaines : {kmf.predict(20):.1%}")
    print(f"Taux de survie a 30 semaines : {kmf.predict(30):.1%}")

    # Dessin du graphique global
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title("Probabilite de rester inscrit (tous les etudiants)")
    plt.xlabel("Nombre de semaines")
    plt.ylabel("Pourcentage d'etudiants")
    plt.grid(True)
    plt.savefig(FIG_DIR / "courbe_globale.png")
    plt.close()

    # --- 2. LES COMPARAISONS (Sous-groupes) ---
    # On regarde les colonnes interessantes
    colonnes_a_comparer = ["boursier", "filiere", "genre", "emploi_parallele"]
    
    for col in colonnes_a_comparer:
        print(f"\nComparaison par : {col}")
        
        # On remplace les valeurs vides par "Inconnu" pour eviter les bugs
        df[col] = df[col].fillna("Inconnu")
        
        plt.figure(figsize=(10, 6))
        
        # On garde uniquement les groupes qui ont au moins 50 etudiants (pour que ce soit fiable)
        valeurs = df[col].value_counts()
        groupes_valides = valeurs[valeurs > 50].index
        
        for groupe in groupes_valides:
            # On filtre les etudiants de ce groupe
            mask = df[col] == groupe
            kmf_groupe = KaplanMeierFitter()
            kmf_groupe.fit(df[mask]["T"], event_observed=df[mask]["E"], label=str(groupe))
            kmf_groupe.plot_survival_function()

        # Test statistique (Log-rank) pour savoir si la difference est reelle ou due au hasard
        try:
            resultat = multivariate_logrank_test(df["T"], df[col], df["E"])
            p_value = resultat.p_value
            print(f"Test du Log-rank (p-value) : {p_value:.4f}")
            if p_value < 0.05:
                print("-> La difference entre les groupes est statistiquement significative.")
            else:
                print("-> Pas de difference notable entre les groupes.")
            plt.title(f"Survie par {col} (p-value = {p_value:.3f})")
        except:
            plt.title(f"Survie par {col}")
            
        plt.xlabel("Nombre de semaines")
        plt.ylabel("Pourcentage d'etudiants")
        plt.grid(True)
        plt.savefig(FIG_DIR / f"courbe_{col}.png")
        plt.close()

    # --- 3. TROUVER LES SEMAINES CRITIQUES ---
    print("\nRecherche des semaines ou il y a le plus d'abandons...")
    # On calcule la baisse de la courbe d'une semaine a l'autre
    courbe = kmf.survival_function_
    courbe["baisse"] = courbe["Tous les etudiants"].diff()
    
    # On prend les 3 plus grosses baisses
    pires_semaines = courbe["baisse"].nsmallest(3)
    for semaine, baisse in pires_semaines.items():
        print(f"  - Semaine {int(semaine)} : chute de {abs(baisse):.1%} d'etudiants")

if __name__ == "__main__":
    analyze_survival()