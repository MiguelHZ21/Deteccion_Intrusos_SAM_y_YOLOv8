# app.py
"""Streamlit front‑end for the Detección de Ladrones demo.

The UI lets the user:
1️⃣ Upload a video (or use a webcam index).
2️⃣ Click **Run** to start the detection pipeline.
3️⃣ View the annotated output video directly in the browser.

The heavy lifting lives in ``detection.py`` which contains the exact same
logic that you already have in the notebook, but now it is wrapped in a
function ``run_detection`` that returns the path to the generated video.
"""

import os
import pathlib
import streamlit as st
from detection import run_detection

st.set_page_config(page_title="Detección de Ladrones", layout="wide")

st.title("🚨 Sistema de Detección de Intenciones (YOLOv8‑seg + SAM)")
st.markdown(
    "Este demo permite subir un video y visualizar las zonas sensibles, el tracking de personas "
    "y el cálculo del *score* de sospecha.  El algoritmo está basado en la "
    "implementación que tenías en el notebook."
)

# ---------------------------------------------------------------------------
# Sidebar – upload / source selection
# ---------------------------------------------------------------------------
source_option = st.sidebar.radio(
    "Fuente de video",
    ("Subir archivo", "Webcam (índice)"),
    index=0,
)

if source_option == "Subir archivo":
    uploaded_file = st.sidebar.file_uploader(
        "Selecciona un video (mp4, avi…)", type=["mp4", "avi", "mov"]
    )
    if uploaded_file is not None:
        # Guardamos el archivo en una carpeta temporal dentro del proyecto
        video_path = pathlib.Path("temp_videos") / uploaded_file.name
        video_path.parent.mkdir(exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
else:
    cam_index = st.sidebar.number_input(
        "Índice de la cámara (0 = predeterminada)", min_value=0, max_value=10, step=1, value=0
    )
    video_path = int(cam_index)  # Streamlit will pass the integer to run_detection

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
if st.sidebar.button("▶️ Ejecutar detección"):
    if isinstance(video_path, pathlib.Path) and not video_path.exists():
        st.error("¡No se encontró el archivo de video!")
    else:
        with st.spinner("Procesando video… Esto puede tardar unos segundos."):
            # Output video will be written to the project root
            output_file = "output_annotated.mp4"
            result_path = run_detection(str(video_path), output_file)
        st.success("¡Procesamiento completado!")
        # Show video player
        st.video(result_path)

st.caption(
    "⚡ La primera ejecución tarda más porque se carga el modelo YOLOv8‑seg. "
    "En ejecuciones posteriores la caché de ``functools.lru_cache`` evita "
    "recargas y la respuesta es mucho más rápida."
)
