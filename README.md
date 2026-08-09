# Portfolio technique — Intelligence artificielle appliquée à la santé

Portfolio de projets individuels explorant l'application de l'intelligence artificielle aux données de santé — aide à la décision clinique, IA explicable, traitement du signal biomédical et vision par ordinateur appliquée à la surveillance de patients.

Ce portfolio est **évolutif** : de nouveaux projets y sont ajoutés au fil de mon parcours de recherche.

## Axes explorés

- Systèmes d'aide à la décision clinique (SIADC)
- Intelligence artificielle explicable (XAI)
- Fusion de données cliniques multimodales
- Traitement du signal biomédical
- Vision par ordinateur appliquée à la santé

## Projets

### [01 — Système explicable d'aide à la décision : prédiction du risque de mortalité en soins intensifs](./01-clinical-decision-support-icu-mortality)

Prédiction du risque de mortalité en soins intensifs à partir de données cliniques multimodales (signes vitaux, laboratoire, descripteurs généraux), avec explicabilité des prédictions par SHAP.

**Résultats clés** : AUROC de 0,887 (XGBoost) · le modèle identifie spontanément le score de Glasgow comme facteur déterminant, cohérent avec les connaissances cliniques établies.

**Outils** : Python, scikit-learn, XGBoost, SHAP

---

### [02 — Vision par ordinateur et traitement du signal pour la surveillance de patients](./02-vision-signal-processing-healthcare)

Deux cas d'usage complémentaires : quantification du mouvement par suivi de points-clés corporels sur vidéo (MediaPipe), et détection/classification de battements cardiaques à partir d'un signal ECG (filtrage, détection de pics, classification).

**Résultats clés** : détection des pics R avec un écart de 0,1 % par rapport à l'annotation experte · classification normal/anormal avec AUROC de 0,999 (limite méthodologique assumée : partage entraînement/test par battement plutôt que par patient — voir le README du projet).

**Outils** : Python, OpenCV, MediaPipe, SciPy, scikit-learn, wfdb

---

*D'autres projets seront ajoutés au fil du temps. Chaque dossier contient son propre README détaillé (méthodologie, résultats, limites assumées).*

## Auteur

Dini Ahamada

