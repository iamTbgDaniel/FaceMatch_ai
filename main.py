import streamlit as st
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
import base64


# Initialisation des modèles
model = InceptionResnetV1(pretrained='vggface2').eval()
detector = MTCNN(image_size=160, margin=20, post_process=True, keep_all=True)

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

def euclidean_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

def select_face(image, label):
    faces = detector(image)
    if faces is None:
        st.error(f"Aucun visage détecté dans {label} !")
        return None

    if faces.ndim == 3 or faces.shape[0] == 1:
        selected = faces.unsqueeze(0) if faces.ndim == 3 else faces
    else:
        st.error(f"{faces.shape[0]} visages détectés dans {label}. L'image doit contenir un seul visage.")
        return None

    with torch.no_grad():
        embedding = model(selected)[0].detach().cpu().numpy()
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
        <li> Calcul de similarité avec distance euclidienne</li>
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

    
    if st.button("Commencer la vérification"):
        st.session_state.page = "Vérification faciale"
        st.rerun() 

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
                distance = euclidean_distance(emb1, emb2)
                st.success(f"Distance euclidienne : {distance:.4f}")

                threshold = 1.0100
                if distance < threshold:
                    st.markdown("<h3 style='color: green;'>Même personne ✅</h3>", unsafe_allow_html=True)
                else:
                    st.markdown("<h3 style='color: red;'>Personnes différentes ❌</h3>", unsafe_allow_html=True)

                with st.expander("Afficher les images"):
                    st.image([image1, image2], width=200)
        else:
            st.warning("Veuillez fournir deux images pour l'analyse.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:white;'>Copyright © 2025 - Application IA FaceMatch | Daniel TSHIBANGU</p>", unsafe_allow_html=True)

