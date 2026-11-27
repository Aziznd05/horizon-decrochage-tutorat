import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration de la page
st.set_page_config(page_title="Horizon - Décrochage", layout="wide")

# Fonction pour charger les données
@st.cache_data
def load_data():
    df = pd.read_csv(Path("data/processed/base_survie.csv"))
    return df

try:
    df = load_data()
    
    # Titre principal
    st.title("Tableau de bord - Prévention du décrochage")
    st.markdown("Bienvenue sur l'outil d'aide à la décision de l'Institut Horizon.")

    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["Etudiants à risque", "Effet du tutorat", "Fiches LLM"])

    # --- ONGLET 1 : LISTE DES ETUDIANTS ---
    with tab1:
        st.header("Liste des étudiants à risque")
        st.write("Voici les étudiants qui ont le plus fort risque de décrochage.")
        
        # Filtres
        filiere_choisie = st.selectbox("Filtrer par filière", ["Toutes"] + list(df["filiere"].dropna().unique()))
        
        df_affiche = df.copy()
        if filiere_choisie != "Toutes":
            df_affiche = df_affiche[df_affiche["filiere"] == filiere_choisie]
            
        # Tri par note (les plus faibles d'abord) pour simuler un risque
        df_affiche = df_affiche.sort_values("note_bac").head(20)
        
        st.dataframe(df_affiche[["id_etudiant", "filiere", "note_bac", "boursier", "T", "E"]])

    # --- ONGLET 2 : EFFET DU TUTORAT ---
    with tab2:
        st.header("L'effet du tutorat")
        st.write("Est-ce que le tutorat est efficace ?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Réduction du risque", value="-12.4%", delta="Significatif")
            st.write("Le tutorat réduit le risque de décrochage.")
            
        with col2:
            st.metric(label="Hazard Ratio (HR)", value="0.876")
            st.write("Un HR < 1 signifie que le tutorat protège.")

        st.info("Pour plus de détails, consultez l'analyse causale dans le rapport technique.")

    # --- ONGLET 3 : FICHES LLM ---
    with tab3:
        st.header("Génération de fiche d'alerte (Simulation)")
        st.write("Sélectionnez un étudiant pour générer une fiche personnalisée.")
        
        # Choix de l'étudiant
        etudiant_choisi = st.selectbox("Choisir un étudiant", df["id_etudiant"].head(50))
        
        if etudiant_choisi:
            info = df[df["id_etudiant"] == etudiant_choisi].iloc[0]
            
            st.subheader(f"Fiche d'alerte pour {info['id_etudiant']}")
            st.write(f"**Filière :** {info['filiere']}")
            st.write(f"**Boursier :** {info['boursier']}")
            st.write(f"**Note au Bac :** {info['note_bac']}")
            
            st.warning("""
            **Fiche d'alerte (Générée par LLM)** :
            Cet étudiant présente un risque de décrochage. 
            Les facteurs principaux sont ses résultats initiaux et sa situation personnelle.
            Nous vous recommandons de lui proposer un entretien de suivi ou de l'orienter vers le tutorat.
            """)
            
            st.caption("Note : Dans un vrai système, ce texte serait généré par une IA (type Llama/Mistral) à partir des chiffres ci-dessus.")

except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")