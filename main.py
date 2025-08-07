import streamlit as st
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import base64


# Initialisation des modèles
model = InceptionResnetV1(pretrained='vggface2').eval() # Modèle de reconnaissance faciale
detector = MTCNN(image_size=160, margin=20, post_process=True, keep_all=True) # Détecteur de visages
if "history" not in st.session_state:
    st.session_state.history = []
# ---------- UTILITAIRES ----------

def add_blurred_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: 100%;
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            filter: blur(7px);
            z-index: -1;
        }}

        .stApp {{
            background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

#def euclidean_distance(emb1, emb2):# Fonction pour calculer la distance euclidienne entre deux embeddings
    #return np.linalg.norm(emb1 - emb2)

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2)  # Produit scalaire entre deux vecteurs unitaires

def select_face(image, label):
    # Détection des visages et des boîtes englobantes
    boxes, _ = detector.detect(image)
    
    if boxes is None or len(boxes) == 0:
        st.error(f"Aucun visage détecté dans {label} !")
        return None

    if len(boxes) > 1:
        st.error(f"{len(boxes)} visages détectés dans {label}. L'image doit contenir un seul visage.")
        return None

    # Récupération de la boîte englobante (xmin, ymin, xmax, ymax)
    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box

    # Recadrage du visage dans l'image originale
    face_crop = image.crop((x1, y1, x2, y2)).resize((160, 160))

    # Affichage du visage recadré
    st.image(face_crop, caption=f"Visage détecté ({label})", width=150)

    # Transformation en tensor pour le modèle
    transform = transforms.ToTensor()
    face_tensor = transform(face_crop).unsqueeze(0)

    # Normalisation de l'embedding
    with torch.no_grad():
        embedding = model(face_tensor)[0].detach().cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)  # ✅ Normalisation ajoutée ici

    return embedding


# ---------- INTERFACE ----------

add_blurred_bg("image3.jpeg")

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Aller à", ["Accueil", "Vérification faciale"])

# ---------- PAGE ACCUEIL ----------
if page == "Accueil":
    st.markdown("<h1 style='text-align: center; color: white;'>Bienvenue sur FaceMatch AI </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #cccccc;'>Vérifiez la similarité entre deux visages grâce à l’intelligence artificielle</h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
        <div style='color: white; font-size: 16px; text-align: justify;'>
            <p><strong>FaceMatch AI</strong> est une application de reconnaissance faciale qui compare deux visages et estime s'ils appartiennent à la même personne. Alimentée par le modèle <strong>FaceNet</strong> et la bibliothèque <strong>facenet-pytorch</strong>, elle fournit une analyse rapide, fiable et précise.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("###  Fonctionnalités principales", unsafe_allow_html=True)
    st.markdown("""
    <ul style='color: white; font-size: 16px;'>
        <li> Chargement d'images ou capture via webcam</li>
        <li> Détection automatique de visage (MTCNN)</li>
        <li> Calcul de similarité avec cosin similarity</li>
        <li> Avertissement en cas d’absence ou de multiples visages</li>
        <li> Interface intuitive et moderne</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("###  Fonctionnement", unsafe_allow_html=True)
    st.markdown("""
    <div style='color: white; font-size: 16px;'>
        <p>1️⃣ Sélectionnez deux photos (ou utilisez la webcam)<br>
        2️⃣ Le système détecte et extrait automatiquement les visages<br>
        3️⃣ Les caracteristiques des visages sont calculées et comparés<br>
        4️⃣ Une distance est affichée avec un verdict clair</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align:center; color:white;'>Copyright © 2025 - Application IA FaceMatch | Daniel TSHIBANGU</p>", unsafe_allow_html=True)


# ---------- PAGE VERIFICATION ----------
elif page == "Vérification faciale":
    st.markdown("<h1 style='text-align: center; color: white;'>VÉRIFICATEUR DE SIMILARITÉ FACIALE AVEC IA [FaceMatch AI]</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:white;'>Téléchargez deux photos ou utilisez votre webcam pour vérifier si ce sont les mêmes personnes.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    image1 = image2 = None

    with col1:
        mode1 = st.selectbox("Méthode pour la première image", ["Uploader", "Webcam"], key="mode1")
        if mode1 == "Uploader":
            uploaded_file1 = st.file_uploader("Image 1", type=["jpg", "jpeg", "png"], key="file1")
            if uploaded_file1:
                image1 = Image.open(uploaded_file1).convert("RGB")
        else:
            picture1 = st.camera_input("Prendre Image 1", key="cam1")
            if picture1:
                image1 = Image.open(picture1).convert("RGB")

    with col2:
        mode2 = st.selectbox("Méthode pour la deuxième image", ["Uploader", "Webcam"], key="mode2")
        if mode2 == "Uploader":
            uploaded_file2 = st.file_uploader("Image 2", type=["jpg", "jpeg", "png"], key="file2")
            if uploaded_file2:
                image2 = Image.open(uploaded_file2).convert("RGB")
        else:
            picture2 = st.camera_input("Prendre Image 2", key="cam2")
            if picture2:
                image2 = Image.open(picture2).convert("RGB")

    if st.button("Analyser"):
        if image1 and image2:
            with st.spinner("Analyse en cours..."):
                emb1 = select_face(image1, "Image 1")
                emb2 = select_face(image2, "Image 2")

            if emb1 is not None and emb2 is not None:
                similarity = cosine_similarity(emb1, emb2)

                # Normalisation du score entre 0 et 1
                similarity_norm = (similarity + 1) / 2  # Convertir de [-1, 1] vers [0, 1]
                #st.progress(similarity_norm)

                # Affichage de la jauge de similarité
                st.markdown("### Score de similarité")
                st.progress(similarity_norm)

                # Couleur du score en fonction du niveau de similarité
                if similarity > 0.85:
                    color = "green"
                elif similarity > 0.67:
                    color = "orange"
                else:
                    color = "red"

                st.markdown(
                    f"<h4 style='color:{color};'>Similarité cosinus : {similarity:.4f}</h4>",
                    unsafe_allow_html=True
                )

                # Verdict
                threshold = 0.67
                if similarity > threshold:
                    st.markdown("<h3 style='color: green;'>Même personne ✅</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: red;'>Personnes différentes ❌</h3>", unsafe_allow_html=True)

                with st.expander("Afficher les images comparées"):
                    st.image([image1, image2], width=200)
                # Historique des comparaisons
                # Stockage de l’historique (images réduites + score + verdict)
            st.session_state.history.append({
                "img1": image1.copy(),
                "img2": image2.copy(),
                "similarity": similarity,
                "verdict": "Même personne ✅" if similarity > threshold else "Personnes différentes ❌"
            })
        else:
            st.warning("Veuillez fournir deux images pour l'analyse.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:white;'>Copyright © 2025 - Application IA FaceMatch | Daniel TSHIBANGU</p>", unsafe_allow_html=True)
# ---------- HISTORIQUE DES COMPARAISONS ----------
if st.session_state.history:
    st.markdown("---")
    st.markdown("### Historique des comparaisons")
    
    for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):  # Limité aux 5 dernières
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.image(entry["img1"], width=100, caption="Image 1")
        with col2:
            st.image(entry["img2"], width=100, caption="Image 2")
        with col3:
            st.markdown(f"**Similarité :** `{entry['similarity']:.4f}`")
            st.markdown(f"**Verdict :** {entry['verdict']}")
        st.markdown("---")