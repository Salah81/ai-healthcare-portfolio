[README.md](https://github.com/user-attachments/files/30878745/README.md)
# Vision par ordinateur et traitement du signal pour la surveillance de patients

Projet individuel de portfolio technique en intelligence artificielle appliquée à la santé, démontrant deux compétences complémentaires : l'analyse vidéo par vision par ordinateur et le traitement de signaux physiologiques.

## Objectif

Ce projet illustre, à travers deux cas d'usage indépendants, comment des méthodes d'IA peuvent transformer des données brutes (vidéo, signal électrophysiologique) en indicateurs cliniquement interprétables :

- **Partie A — Vision par ordinateur** : suivi de points-clés corporels sur vidéo pour quantifier automatiquement le mouvement d'une personne dans le temps — une approche transposable à l'évaluation de la motilité ou de l'agitation de patients à partir de vidéo.
- **Partie B — Traitement du signal** : filtrage, détection automatique de pics et classification de battements cardiaques à partir d'un signal ECG — une approche transposable à la surveillance continue de signes vitaux.

## Partie A — Suivi de mouvement par vision par ordinateur

**Méthode** : détection de 33 points-clés corporels par image (MediaPipe Pose Landmarker), puis calcul d'un indice d'agitation basé sur le déplacement moyen des points-clés d'une image à l'autre.

**Résultat** : la courbe d'agitation obtenue reflète fidèlement les phases de mouvement et d'immobilité relative de la séquence vidéo analysée, démontrant la viabilité de l'approche pour une quantification automatique et continue du mouvement.

**Limites** : la vidéo utilisée est une séquence générique (non clinique) ; un contexte réel de surveillance de patients impliquerait des caméras fixes (RVB-D, infrarouge) et des conditions d'éclairage et de cadrage contrôlées, différentes de celles testées ici.

## Partie B — Analyse de signal ECG

**Données** : MIT-BIH Arrhythmia Database (PhysioNet), jeu de données ouvert de référence en analyse de signaux cardiaques, combinant 8 enregistrements pour un total de 17 813 battements annotés.

**Méthode** :
1. Filtrage passe-bande (0,5–40 Hz) pour éliminer le bruit et la dérive de ligne de base.
2. Détection automatique des pics R par recherche de maxima locaux.
3. Extraction de caractéristiques par battement (intervalles RR, amplitude, variabilité locale).
4. Classification normal / anormal par forêt aléatoire.

**Résultats** :
- Détection des pics R : 2 277 pics détectés automatiquement contre 2 274 battements annotés par des experts sur l'enregistrement de référence (écart de 0,1 %).
- Classification : AUROC de 0,999, rappel de 0,98 sur la classe anormale.

**Limite méthodologique assumée** : le partage entraînement/test a été effectué par battement plutôt que par patient, ce qui signifie que des battements d'un même patient peuvent apparaître à la fois dans l'ensemble d'entraînement et de test. Ce choix, courant mais optimiste, tend à surestimer la performance de généralisation à de nouveaux patients — un biais bien documenté dans la littérature sur la classification de signaux ECG. Une évaluation par validation croisée au niveau du patient (« patient-independent split ») donnerait une estimation plus réaliste et constitue la prochaine étape naturelle de ce travail.

## Pistes futures

- Validation croisée par patient plutôt que par battement, pour une estimation réaliste de la généralisation.
- Remplacement de l'extraction de caractéristiques manuelle par un modèle profond (CNN 1D) appliqué directement au signal brut.
- Extension de l'analyse vidéo à des données RVB-D ou infrarouges, plus robustes aux conditions d'éclairage variables.
- Fusion des deux modalités (vidéo + signal) dans un même pipeline d'analyse multimodale.

## Outils utilisés

Python · OpenCV · MediaPipe (Pose Landmarker) · SciPy · scikit-learn · wfdb · Google Colab

## Exécution

Le notebook est autonome et conçu pour Google Colab : ouvrir `Vision_Signal_portfolio.ipynb`, exécuter les cellules dans l'ordre. La partie vidéo nécessite le téléversement d'un court clip (10-20 secondes) ; la partie signal télécharge automatiquement les données depuis PhysioNet.

## Auteur

Dini Ahamada
