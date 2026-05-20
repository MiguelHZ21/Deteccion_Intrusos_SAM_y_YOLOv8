# detection.py
"""Detection utilities for the Detección de Ladrones Streamlit app.

This module extracts the core logic from the Jupyter notebook so that it can be
imported by a plain Python script (app.py).  The function `run_detection`
processes a video source, draws the zones, tracks people and returns the path
to the annotated output video.

All heavy‑weight objects (YOLO model, zone data) are cached with
`functools.lru_cache` to avoid re‑loading on every Streamlit run.
"""

import os
import json
import cv2
import numpy as np
import torch
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from functools import lru_cache

# ---------------------------------------------------------------------------
# Configuration constants (mirrored from the notebook). Adjust only here.
# ---------------------------------------------------------------------------
MIN_DWELL_IN_SEC = 1.0          # seconds a person must stay inside a zone
MIN_OUT_SEC = 0.5               # tolerance for short loss of detection (seconds)
VENTANA_VISITAS_SEC = 120.0
VENTANA_TRAJ_SEC = 15.0
UMBRAL_RATIO = 4.0
UMBRAL_AREA_PX = 150000
SCORE_ALERTA = 10.0

# Weight per zone colour
ZONA_WEIGHTS = {"amarillo": 1.0, "naranja": 2.0, "rojo": 3.0}
colores_ui = {"amarillo": (0, 255, 255), "naranja": (0, 165, 255), "rojo": (0, 0, 255)}

# ---------------------------------------------------------------------------
# Helper classes – unchanged from notebook (only minimal docstrings added).
# ---------------------------------------------------------------------------
class TrackState:
    """State kept per tracked person (identified by YOLO track ID)."""
    def __init__(self, track_id: int):
        self.track_id = track_id
        self.history = deque()
        self.smoothed_pt = None
        self.current_zone_color = None
        self.zone_entry_frame = 0
        self.last_out_frame = -9999
        self.visitas = []  # list of dicts {"color": str, "frame": int}
        self.es_sospechoso = False

def get_horario_weight() -> float:
    """Return a multiplier based on the current hour of the day."""
    h = datetime.now().hour
    if 6 <= h < 20:
        return 1.0
    elif 20 <= h < 23:
        return 1.3
    else:
        return 1.8

# ---------------------------------------------------------------------------
# Cached model and zone loading – these are expensive, so we keep them memoised.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_yolo_model() -> YOLO:
    """Load the YOLOv8‑seg model (tiny variant) once per process."""
    modelo = YOLO("yolov8n-seg.pt")
    return modelo

@lru_cache(maxsize=1)
def load_zonas(path: str = "calibracion_zonas.json") -> list:
    """Load the calibrated zones from JSON (or return empty list)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    zonas = []
    for z in data.get("zones", []):
        color = z.get("color", "amarillo") if isinstance(z, dict) else "amarillo"
        pts = z.get("points", z) if isinstance(z, dict) else z
        zonas.append({"color": color, "contour": np.array(pts, dtype=np.int32)})
    return zonas

# ---------------------------------------------------------------------------
# Core detection routine – returns the path to an output video file.
# ---------------------------------------------------------------------------
def run_detection(source_path: str, output_path: str = "output.mp4") -> str:
    """Process *source_path* frame‑by‑frame, annotate and write *output_path*.

    Parameters
    ----------
    source_path: str
        Path to the input video (or integer index for a webcam).
    output_path: str, optional
        Where to store the annotated video.  The file is overwritten each run.

    Returns
    -------
    str
        The absolute path to the generated video.
    """
    modelo_yolo = load_yolo_model()
    zonas_sensibles = load_zonas()

    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    track_states = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        resultados = modelo_yolo.track(frame, persist=True, classes=[0], tracker="botsort.yaml", verbose=False)
        annotated = resultados[0].plot()
        for z in zonas_sensibles:
            contour = z["contour"]
            color = colores_ui[z["color"]]
            cv2.drawContours(annotated, [contour], -1, color, 2)
            overlay = annotated.copy()
            cv2.drawContours(overlay, [contour], -1, color, -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        if resultados[0].boxes is not None and resultados[0].boxes.id is not None:
            boxes = resultados[0].boxes.xyxy.cpu().numpy()
            ids = resultados[0].boxes.id.cpu().numpy()
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = y1  # top‑of‑bbox (head)
                if tid not in track_states:
                    track_states[tid] = TrackState(tid)
                ts = track_states[tid]
                # smoothing
                if ts.smoothed_pt is None:
                    ts.smoothed_pt = (cx, cy)
                else:
                    ts.smoothed_pt = (
                        0.8 * ts.smoothed_pt[0] + 0.2 * cx,
                        0.8 * ts.smoothed_pt[1] + 0.2 * cy,
                    )
                ts.history.append((ts.smoothed_pt[0], ts.smoothed_pt[1], frame_count))
                limite = frame_count - int(VENTANA_TRAJ_SEC * fps)
                while ts.history and ts.history[0][2] < limite:
                    ts.history.popleft()
                # zone hysteresis (more tolerant)
                zona_tocada_ahora = None
                for c_test in ["rojo", "naranja", "amarillo"]:
                    max_dist = -9999
                    for z in zonas_sensibles:
                        if z["color"] != c_test:
                            continue
                        d = cv2.pointPolygonTest(z["contour"], (cx, cy), True)
                        if d > max_dist:
                            max_dist = d
                    if ts.current_zone_color == c_test:
                        if max_dist >= -30:
                            zona_tocada_ahora = c_test
                            break
                    else:
                        if max_dist >= -15:
                            zona_tocada_ahora = c_test
                            break
                # state transition
                if zona_tocada_ahora != ts.current_zone_color:
                    frames_out = frame_count - ts.last_out_frame
                    if frames_out >= int(MIN_OUT_SEC * fps) or ts.current_zone_color is not None:
                        if ts.current_zone_color is not None:
                            frames_in = frame_count - ts.zone_entry_frame
                            if frames_in >= int(MIN_DWELL_IN_SEC * fps):
                                ts.visitas.append({"color": ts.current_zone_color, "frame": frame_count})
                        ts.current_zone_color = zona_tocada_ahora
                        if zona_tocada_ahora is not None:
                            ts.zone_entry_frame = frame_count
                        else:
                            ts.last_out_frame = frame_count
                # purge old visits
                lim = frame_count - int(VENTANA_VISITAS_SEC * fps)
                ts.visitas = [v for v in ts.visitas if v["frame"] >= lim]
                # scoring
                score = 0.0
                for v in ts.visitas:
                    score += ZONA_WEIGHTS[v["color"]]
                if ts.current_zone_color is not None:
                    segs = (frame_count - ts.zone_entry_frame) / fps
                    w = ZONA_WEIGHTS[ts.current_zone_color]
                    if segs > 3.0:
                        score += w * 1.0
                    if segs > 8.0:
                        score += w * 1.5
                    if segs > 15.0:
                        score += w * 2.0
                if len(ts.history) > 10:
                    pts = np.array([(p[0], p[1]) for p in ts.history])
                    dist_total = np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1)))
                    dist_neta = np.sqrt(np.sum((pts[-1] - pts[0]) ** 2))
                    ratio = dist_total / (dist_neta + 1e-5)
                    minx, miny = np.min(pts, axis=0)
                    maxx, maxy = np.max(pts, axis=0)
                    area_traj = (maxx - minx) * (maxy - miny)
                    if ratio >= UMBRAL_RATIO:
                        score += 3
                    if 0 < area_traj < UMBRAL_AREA_PX:
                        score += 2
                w_horario = get_horario_weight()
                if len(track_states) > 3:
                    w_horario *= 0.6
                score *= w_horario
                if score >= SCORE_ALERTA:
                    ts.es_sospechoso = True
                # UI overlay
                box_color = (0, 0, 255) if ts.es_sospechoso else (0, 255, 0)
                cv2.circle(annotated, (cx, cy), 5, box_color, -1)
                if len(ts.history) > 1:
                    pts_draw = np.array([(int(p[0]), int(p[1])) for p in ts.history])
                    cv2.polylines(annotated, [pts_draw], False, (255, 255, 0), 2)
                if ts.es_sospechoso:
                    cv2.putText(
                        annotated,
                        f"ALERTA (S:{score:.1f})",
                        (x1, max(30, y1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3,
                    )
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                else:
                    cv2.putText(
                        annotated,
                        f"S:{score:.1f} V:{len(ts.visitas)}",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
        out.write(annotated)
    cap.release()
    out.release()
    return os.path.abspath(output_path)
