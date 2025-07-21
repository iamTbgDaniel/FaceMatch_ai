# FaceMatch AI

**FaceMatch AI** est une application intelligente de reconnaissance faciale, développée avec **Streamlit** et alimentée par le modèle **FaceNet** via la bibliothèque `facenet-pytorch`. Elle permet de comparer deux visages et d’estimer s’ils appartiennent à la même personne.

---

##  Fonctionnalités principales

- Téléversement ou capture d’image via **webcam**
- Détection automatique de visages avec **MTCNN**
- Calcul de similarité via **distance euclidienne**
- Alerte en cas de visage manquant ou de plusieurs visages
- Interface utilisateur fluide et responsive grâce à Streamlit

---

##  Interface

L’application comporte deux pages principales accessibles via une barre latérale :

- **Accueil** : Présentation, fonctionnement, bouton d’accès rapide
- **Vérification faciale** : Téléversement ou capture de deux images, détection, analyse et verdict

---

##  Technologies utilisées

- [Python 3.11+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [Torch](https://pytorch.org/)
- [FaceNet via facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [MTCNN](https://github.com/ipazc/mtcnn)
- [Pillow](https://python-pillow.org/)

---

## Installation

1. Clone ce dépôt :

```bash
git clone https://github.com/iamTbgDaniel/FaceMatch_ai.git
cd FaceMatch_ai

2. Installe les dépendances:

```bash
pip install -r requirements.txt

3. Lance l'application:

```bash
streamlit run main.py

---

## Exemple de fonctionnement

- L'utilisateur charge deux images ou utilise la webcam

- L’application détecte automatiquement les visages

- Un score de similarité (distance euclidienne) est    affiché

- Si la distance est inférieure à 1.0111, les visages sont considérés comme identiques

Auteur:
Daniel TSHIBANGU
Projet réalisé dans le cadre du programme L3 Data Science - 2025
GitHub : @iamTbgDaniel

📄 Licence
Ce projet est open-source et sous licence MIT.