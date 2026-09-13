# Portfolio technique — Intelligence artificielle appliquée à la santé

Portfolio de projets individuels explorant l'application de l'intelligence artificielle aux données de santé — aide à la décision clinique, IA explicable, traitement du signal biomédical, vision par ordinateur, deep learning pour l'imagerie médicale, segmentation tumorale, analyse quantitative d'images et visualisation 3D / réalité augmentée pour l'imagerie médicale.

Ce portfolio est **évolutif** : de nouveaux projets y sont ajoutés au fil de mon parcours de recherche.

## Axes explorés

- Systèmes d'aide à la décision clinique (SIADC)
- Intelligence artificielle explicable (XAI)
- Fusion de données cliniques multimodales
- Traitement du signal biomédical
- Vision par ordinateur appliquée à la santé
- Visualisation 3D et réalité augmentée pour l'imagerie médicale
- Deep learning pour l'imagerie médicale
- Segmentation tumorale et analyse quantitative d'images médicales

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

### [03 — Pipeline de réalité augmentée pour l'imagerie par ultrasons : preuve de concept](./03-ar-ultrasound-imaging)

Pipeline complète couvrant les trois composantes d'un système de RA pour l'échographie : segmentation de structures anatomiques sur un phantom 3D synthétique, visualisation 3D par isosurfaces (VTK/PyVista), et simulation d'un overlay AR guidant le positionnement de la sonde.

**Résultats clés** : segmentation validée quantitativement contre une vérité terrain (Dice de 0,99 pour le foie, 0,87 pour la lésion, 0,86 pour le vaisseau) · rendu 3D par isosurfaces des structures segmentées (limite méthodologique assumée : suivi de sonde paramétrique plutôt que dérivé d'un algorithme de tracking réel — voir le README du projet).

**Outils** : Python, scikit-image, OpenCV, PyVista/VTK

---

### [04 — LiverAI : segmentation de tumeurs hépatiques par deep learning et analyse de résultats](https://github.com/Salah81/ai-healthcare-portfolio/tree/main/04-liver-tumor-deep-learning)

Pipeline de recherche reproductible reliant **segmentation tumorale par U-Net**, analyse quantitative de la tumeur et modélisation prédictive au niveau patient. Le projet comprend un générateur déterministe d'images CT synthétiques, une architecture U-Net 2D en PyTorch, l'extraction de caractéristiques morphologiques et d'intensité, ainsi qu'un pipeline de prédiction de résultats cliniques servant de preuve de concept méthodologique.

**Résultats clés** : segmentation sur données synthétiques avec **Dice de 0,892 ± 0,052**, **IoU de 0,808** et **HD95 de 1,0 pixel** · pipeline de prédiction aval fonctionnel avec **AUROC de 0,937** et **accuracy de 0,813** sur des labels synthétiques. Ces résultats valident la reproductibilité et le fonctionnement du pipeline logiciel, mais **ne constituent pas des performances cliniques**.

Le projet inclut également une visualisation détaillée des **vrais positifs, faux positifs et faux négatifs**, une courbe d'entraînement reproductible, des tests automatisés et une intégration continue GitHub Actions.

**Orientation recherche** : extension vers des données CT hépatiques réelles, segmentation 3D/2.5D, combinaison de représentations profondes et de caractéristiques radiomiques, puis modélisation longitudinale de la progression tumorale.

**Outils** : Python, PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, pytest, GitHub Actions

---

*D'autres projets seront ajoutés au fil du temps. Chaque dossier contient son propre README détaillé (méthodologie, résultats, limites assumées).*

## Auteur

Dini Ahamada
