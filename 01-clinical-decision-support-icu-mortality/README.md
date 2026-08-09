[README.md](https://github.com/user-attachments/files/30878708/README.md)
# Système explicable d'aide à la décision — Prédiction du risque de mortalité en soins intensifs

Projet individuel de portfolio technique, réalisé dans le cadre d'une candidature au doctorat en génie, axée sur l'intelligence artificielle appliquée à la santé numérique.

## Objectif

Construire un modèle de prédiction du risque de mortalité en soins intensifs à partir de données cliniques multimodales (signes vitaux, résultats de laboratoire, descripteurs généraux), puis **expliquer** chaque prédiction avec SHAP — afin de produire une sortie interprétable, proche de ce qu'un système d'aide à la décision clinique (SIADC) devrait fournir à un professionnel de la santé.

Ce projet illustre trois axes de recherche qui m'intéressent particulièrement :
- la **fusion de données cliniques multimodales**,
- la **prédiction du risque** par apprentissage automatique,
- et surtout l'**explicabilité clinique** des prédictions, pour produire une recommandation interprétable plutôt qu'un score opaque.

## Jeu de données

**PhysioNet/Computing in Cardiology Challenge 2012** — *Predicting Mortality of ICU Patients* (Silva et al., 2012).
4 000 séjours en soins intensifs adultes (Training Set A), avec :
- 48 heures de séries temporelles cliniques (37 variables : fréquence cardiaque, pression artérielle, créatinine, glucose, etc.)
- 5 descripteurs généraux (âge, sexe, taille, poids, type d'unité de soins)
- Une étiquette binaire : décès intra-hospitalier ou non

Jeu de données en accès libre, sans processus d'accréditation.

## Méthodologie

1. **Agrégation des séries temporelles** — chaque variable clinique est résumée par patient (min, max, moyenne, dernière valeur mesurée sur 48h)
2. **Modélisation** — comparaison de deux modèles : Random Forest (référence) et XGBoost
3. **Évaluation** — AUROC, précision/rappel, matrice de confusion
4. **Explicabilité (SHAP)** — importance globale des variables + explication individuelle par patient
5. **Traduction en langage clair** — transformation des valeurs SHAP en recommandation textuelle simple, destinée à un professionnel de la santé

## Résultats

| Modèle | AUROC | Rappel (classe Décès) |
|---|---|---|
| Random Forest | 0,863 | 0,58 |
| **XGBoost** | **0,887** | 0,57 |

**Validation clinique du modèle** : l'analyse SHAP identifie spontanément le score de Glasgow (GCS) comme le facteur le plus déterminant dans les prédictions — un indicateur de gravité largement reconnu en soins intensifs (composante centrale de scores cliniques établis comme l'APACHE ou le SAPS). Cette cohérence avec les connaissances médicales existantes suggère que le modèle capte un signal clinique réel, plutôt qu'une corrélation statistique fortuite.

## Limites assumées

- Population adulte, et non pédiatrique
- Analyse rétrospective sur données statiques, plutôt qu'un traitement en temps réel
- Agrégation simple des séries temporelles (min/max/moyenne), plutôt qu'une modélisation séquentielle (LSTM, Transformer) — un choix délibéré pour un prototype réalisable en quelques jours

## Pistes futures

- Modélisation séquentielle des séries temporelles plutôt que leur agrégation
- Interface interactive (Streamlit/Gradio) pour simuler un usage clinique
- Extension à des données multimodales incluant de l'imagerie

## Outils utilisés

Python · pandas · scikit-learn · XGBoost · SHAP · Google Colab

## Exécution

Le notebook est autonome et conçu pour tourner sur Google Colab : ouvrir `notebook.ipynb`, exécuter les cellules dans l'ordre (le téléchargement des données et l'installation des dépendances sont inclus dans le notebook).

## Auteur

Dini Ahamada
