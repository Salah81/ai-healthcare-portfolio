"""
04_ar_probe_simulation.py
---------------------------
Simulation d'un overlay AR pour guidage échographique.
Simule:
  1. Une caméra regardant un patient (image RGB)
  2. Un probe US virtuel dont la position est trackée
  3. La projection de l'image US 2D sur le patient (overlay AR)
  
C'est l'équivalent logiciel de ce que ferait un système AR avec:
  - Tracking du probe (optique ou EM)
  - Calibration caméra-probe
  - Rendu de la slice US dans le repère patient
"""
import os
import numpy as np
import cv2

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_patient_background(width=800, height=600):
    """
    Simule une image de caméra regardant un torse patient.
    En réalité, ce serait le flux vidéo d'une caméra AR (HoloLens, etc.).
    """
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    # Peau
    bg[100:500, 150:650] = [210, 180, 160]
    # Ombres
    cv2.ellipse(bg, (400, 300), (200, 120), 0, 0, 360, (180, 150, 130), -1)
    # Marqueurs anatomiques (ombilic, côtes)
    cv2.circle(bg, (400, 320), 8, (160, 130, 110), -1)
    cv2.line(bg, (250, 220), (550, 220), (190, 160, 140), 2)
    cv2.line(bg, (260, 260), (540, 260), (190, 160, 140), 2)
    return bg

def get_us_slice_from_volume(volume, angle_deg=15, slice_z=64):
    """
    Extrait une slice 2D du volume 3D avec rotation.
    Simule l'acquisition d'une image US par le probe orienté.
    """
    h, w = volume.shape[0], volume.shape[1]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    slice_2d = volume[:, :, slice_z]
    rotated = cv2.warpAffine(slice_2d, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated

def apply_us_colormap(slice_gray):
    """
    Applique une fausse couleur typique de l'échographie médicale
    (souvent du jaune-vert ou grayscale inversé selon les appareils).
    """
    slice_8bit = (slice_gray * 255).astype(np.uint8)
    # Colormap personnalisée: noir -> bleu -> cyan -> jaune -> blanc
    colored = cv2.applyColorMap(slice_8bit, cv2.COLORMAP_TURBO)
    return colored

def composite_ar_frame(patient_bg, us_image, probe_pos, probe_angle):
    """
    Compose l'image finale AR:
      - Fond: caméra patient
      - Overlay: image US projetée à la position du probe
      - HUD: informations cliniques (position, angle, profondeur)
    """
    frame = patient_bg.copy()
    h, w = frame.shape[:2]

    # Redimensionnement de l'image US pour qu'elle s'intègre au torse
    us_h, us_w = 200, 160
    us_resized = cv2.resize(us_image, (us_w, us_h))

    # Position du coin supérieur gauche de l'overlay
    x, y = probe_pos

    # Création d'un masque pour l'overlay (forme arrondie de sonde)
    mask = np.zeros((us_h, us_w), dtype=np.uint8)
    cv2.ellipse(mask, (us_w//2, us_h//2), (us_w//2 - 5, us_h//2 - 5),
                probe_angle, 0, 360, 255, -1)

    # Rotation de l'overlay selon l'angle du probe
    if probe_angle != 0:
        center = (us_w // 2, us_h // 2)
        M = cv2.getRotationMatrix2D(center, -probe_angle, 1.0)
        us_resized = cv2.warpAffine(us_resized, M, (us_w, us_h),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        mask = cv2.warpAffine(mask, M, (us_w, us_h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Fusion avec le fond (alpha blending)
    x1, y1 = max(0, x - us_w//2), max(0, y - us_h//2)
    x2, y2 = min(w, x1 + us_w), min(h, y1 + us_h)

    # Ajustement si hors limites
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return frame  # skip si hors cadre

    roi = frame[y1:y2, x1:x2]
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0

    # Blend: AR overlay semi-transparent
    blended = (roi * (1 - mask_3ch * 0.75) + us_resized * (mask_3ch * 0.75)).astype(np.uint8)
    frame[y1:y2, x1:x2] = blended

    # --- HUD (Head-Up Display) AR ---
    # Cadre autour du probe
    cv2.ellipse(frame, (x, y), (us_w//2 + 10, us_h//2 + 10),
                -probe_angle, 0, 360, (0, 255, 255), 2)

    # Ligne de pointillés vers la profondeur (simulation)
    cv2.line(frame, (x, y + us_h//2 + 10), (x, y + 180), (0, 255, 255), 1, cv2.LINE_AA)

    # Texte d'information
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"PROBE: {probe_angle:.1f} deg", (x + 60, y - 40),
                font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"DEPTH: 8.5 cm", (x + 60, y - 20),
                font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "MODE: B-Mode", (x + 60, y),
                font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Croix centrale (crosshair)
    cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

    return frame

def main():
    print("AR Ultrasound Probe Simulation")

    # Chargement du volume
    volume = np.load(f"{DATA_DIR}/phantom_us_3d.npy")

    # Création du fond patient
    patient_bg = create_patient_background(800, 600)

    # Paramètres de la simulation
    probe_positions = [(350, 280), (400, 300), (450, 320), (400, 350)]
    probe_angles = [10, 15, 20, 12]
    slice_indices = [60, 64, 68, 62]

    for i, (pos, angle, sidx) in enumerate(zip(probe_positions, probe_angles, slice_indices)):
        # Extraction de la slice US
        us_slice = get_us_slice_from_volume(volume, angle_deg=angle, slice_z=sidx)
        us_colored = apply_us_colormap(us_slice)

        # Composition AR
        frame = composite_ar_frame(patient_bg, us_colored, pos, angle)

        # Sauvegarde
        out_path = f"{RESULTS_DIR}/04_ar_frame_{i+1:02d}.png"
        cv2.imwrite(out_path, frame)
        print(f"  Saved: {out_path}")

    # Image finale: collage des 4 frames
    frames = [cv2.imread(f"{RESULTS_DIR}/04_ar_frame_{i+1:02d}.png") for i in range(4)]
    top = np.hstack(frames[:2])
    bottom = np.hstack(frames[2:])
    collage = np.vstack([top, bottom])
    collage_rgb = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 10))
    plt.imshow(collage_rgb)
    plt.axis('off')
    plt.title("AR Ultrasound Simulation: Probe Tracking & In-Situ Overlay",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/04_ar_simulation_collage.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR}/04_ar_simulation_collage.png")
    plt.show()

if __name__ == "__main__":
    main()