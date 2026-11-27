# Horizon - Anticiper le décrochage et mesurer l'effet du tutorat

**Étudiant** : Abdou Aziz NDIAYE
**Projet** : Institut Supérieur Horizon (Expert IA & LLM)

## Présentation du projet
Ce projet vise à analyser le décrochage étudiant à l'aide de données d'apprentissage. Il se divise en trois parties :
1. **Nettoyage et analyse de survie** : Prédire *quand* les étudiants risquent de décrocher.
2. **Inférence causale** : Prouver si le tutorat *réduit* réellement le décrochage (en corrigeant le biais de sélection).
3. **Outil d'aide à la décision** : Un tableau de bord (Streamlit) avec génération de fiches d'alerte.

## Installation et Lancement

### Prérequis
- Python 3.11 (via Anaconda recommandé)
- Git

### Installation
```bash
# 1. Créer l'environnement virtuel (avec conda)
conda create -n horizon python=3.11 -y
conda activate horizon

# 2. Installer les dépendances
pip install -r requirements.txt