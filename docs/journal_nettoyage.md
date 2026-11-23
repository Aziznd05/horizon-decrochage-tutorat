## Table `inscriptions.csv`

| Anomalie | Détail | Traitement appliqué |
|---|---|---|
| Séparateur de fichier | Utilisation de `;` au lieu de `,` | `pd.read_csv(sep=";")` |
| Dates hétérogènes | `2025-10-06`, `06/10/2025`, `7 oct. 2024`, `31 oct. 2024` | Remplacement des mois français, puis `pd.to_datetime(dayfirst=True)` |
| Statuts hétérogènes | `inscrit`, `INSCRIT`, `abandonné`, `transfere`, `ABANDON` | Passage en minuscules, suppression des accents, mapping vers 3 statuts (`inscrit`, `abandon`, `transfert`) |
| Incohérences temporelles | Date de fin antérieure à la date de début (ex: `HZ29974`) | `date_fin` mise à `NaT` |
| Doublons | `HZ92160`, `HZ16673`, `HZ77451` apparaissent deux fois | Suppression en gardant la 1ère occurrence |

## Table `tutorat.csv`

| Anomalie | Détail | Traitement appliqué |
|---|---|---|
| Nom de colonne | `ID Etudiant` | Renommé en `id_etudiant` |
| Identifiants | Minuscules, espaces | `str.strip().str.upper()` |
| Semaines aberrantes | `52`, `45`, `25`, `0` | Valeurs hors [1, 30] -> `NaN` (incohérence temporelle) |
| Nb séances invalide | `-2.0`, `0` | Valeurs négatives -> `NaN` |
| Tuteurs hétérogènes | `pair (étudiant avancé)`, `PAIR`, `Prof`, `ENSEIGNANT` | Mapping vers `Pair`, `Enseignant`, `Étudiant` |
| Doublons stricts | `HZ70667`, `HZ90929`, `HZ56306` | Suppression sur `(id_etudiant, semaine_debut)` |