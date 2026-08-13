"""
02_preprocess_segment.py
------------------------
Pipeline de prétraitement d'image échographique:
  - Réduction du speckle (filtre médian)
  - Amélioration du contraste (CLAHE)
  - Segmentation des structures d'intérêt
"""
import os
import numpy as np
import cv2
from scipy import ndimage
from skimage.filters import threshold_multiotsu
import matplotlib.pyplot as plt

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def apply_clahe_3d(volume):
    """CLAHE slice-par-slice -- utilise UNIQUEMENT pour l'affichage."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = np.zeros_like(volume)
    for i in range(volume.shape[2]):
        slice_2d = (volume[:, :, i] * 255).astype(np.uint8)
        enhanced[:, :, i] = clahe.apply(slice_2d) / 255.0
    return enhanced


def largest_component(mask, min_size=1):
    """Garde uniquement la plus grande composante connexe (retire le bruit poivre-et-sel)."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    if sizes.max() < min_size:
        return np.zeros_like(mask)
    biggest = np.argmax(sizes) + 1
    return labeled == biggest


def largest_compact_component(mask, min_size=1, max_aspect=3.0):
    """
    Comme largest_component, mais rejette les composantes "en nappe fine"
    (ex: la paroi abdominale) en comparant les dimensions de leur bounding
    box. Une lesion est ~spherique (aspect proche de 1) ; une nappe fine
    a un aspect tres eleve (une dimension beaucoup plus petite que les
    deux autres). Retombe sur largest_component si aucune composante
    compacte n'est trouvee.
    """
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    objs = ndimage.find_objects(labeled)
    best_label, best_size = None, 0
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        size = (labeled[sl] == i).sum()
        if size < min_size:
            continue
        dims = [sl[a].stop - sl[a].start for a in range(3)]
        aspect = max(dims) / max(1, min(dims))
        if aspect <= max_aspect and size > best_size:
            best_label, best_size = i, size
    if best_label is None:
        return largest_component(mask, min_size=min_size)
    return labeled == best_label


def filter_by_size(mask, min_size):
    """Retire toutes les composantes connexes plus petites que min_size (nettoyage du bruit)."""
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.zeros_like(mask)
    for i, s in enumerate(sizes, start=1):
        if s >= min_size:
            keep |= (labeled == i)
    return keep


def segment_structures(denoised_vol):
    """
    Segmentation par seuillage Otsu multi-niveaux + nettoyage morphologique,
    calibree sur le volume DEBRUITE (pas CLAHE).
    """
    fg_values = denoised_vol[denoised_vol > 0.01]
    # 4 classes sur le foreground: vaisseau (bas) / foie / paroi-lesion (haut)
    t1, t2, t3 = threshold_multiotsu(fg_values, classes=4)

    body = denoised_vol > 0.01
    body_filled = ndimage.binary_fill_holes(body)          # interieur "plein" du torse
    body_filled = largest_component(body_filled)             # retire les artefacts isoles

    # Vaisseau: valeur basse, a l'interieur du corps (pas le fond), on garde
    # la plus grande composante connexe (le vaisseau est le seul objet sombre
    # de cette taille -- le reste n'est que du bruit speckle residuel)
    vessel_candidate = (denoised_vol > 0.01) & (denoised_vol <= t1) & body_filled
    vessel = largest_component(vessel_candidate, min_size=150)
    vessel = ndimage.binary_closing(vessel, iterations=1)

    # Lesion: valeur haute + forme compacte (spherique), pas une nappe fine
    # (la paroi abdominale depasse aussi ce seuil par endroits, mais elle est
    # plate: on l'elimine avec le filtre d'aspect-ratio)
    lesion_candidate = denoised_vol > t3
    lesion = largest_compact_component(lesion_candidate, min_size=80, max_aspect=3.0)
    lesion = ndimage.binary_closing(lesion, iterations=1)

    # Foie: tout le tissu "plein" du corps, hors vaisseau et hors lesion
    liver = body_filled & ~vessel & ~lesion
    liver = ndimage.binary_closing(liver, iterations=1)

    return liver, lesion, vessel, (t1, t2, t3)


def dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return 2 * inter / denom if denom > 0 else 1.0


if __name__ == "__main__":
    print("Loading phantom + ground-truth masks...")
    phantom = np.load(f"{DATA_DIR}/phantom_us_3d.npy")
    gt_liver = np.load(f"{DATA_DIR}/gt_liver_mask.npy")
    gt_vessel = np.load(f"{DATA_DIR}/gt_vessel_mask.npy")
    gt_lesion = np.load(f"{DATA_DIR}/gt_lesion_mask.npy")
    gt_liver_only = gt_liver & ~gt_vessel  # tissu hepatique hors lumiere du vaisseau

    print("Denoising (median filter 3D)...")
    denoised = ndimage.median_filter(phantom, size=3)

    print("Enhancing for DISPLAY only (CLAHE)...")
    enhanced = apply_clahe_3d(denoised)

    print("Segmenting structures (Otsu multi-niveaux sur volume debruite)...")
    liver_mask, lesion_mask, vessel_mask, thresholds = segment_structures(denoised)
    print(f"  Seuils Otsu calibres automatiquement: {[round(t,3) for t in thresholds]}")

    d_liver = dice(liver_mask, gt_liver_only)
    d_lesion = dice(lesion_mask, gt_lesion)
    d_vessel = dice(vessel_mask, gt_vessel)
    print(f"  Dice foie   : {d_liver:.3f}")
    print(f"  Dice lesion : {d_lesion:.3f}")
    print(f"  Dice vaisseau: {d_vessel:.3f}")

    np.save(f"{DATA_DIR}/phantom_denoised.npy", denoised)
    np.save(f"{DATA_DIR}/liver_mask.npy", liver_mask)
    np.save(f"{DATA_DIR}/lesion_mask.npy", lesion_mask)
    np.save(f"{DATA_DIR}/vessel_mask.npy", vessel_mask)
    print("Segmentation masks saved.")

    # Visualisation
    slice_idx = 64
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(phantom[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0, 0].set_title("Original (Speckle)")

    axes[0, 1].imshow(denoised[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0, 1].set_title("Denoised (Median) — used for segmentation")

    axes[0, 2].imshow(enhanced[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0, 2].set_title("Enhanced (CLAHE) — display only")

    axes[1, 0].imshow(phantom[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 0].imshow(liver_mask[:, :, slice_idx].T, cmap='Greens', origin='lower', alpha=0.6)
    axes[1, 0].set_title(f"Liver Segmentation (Dice={d_liver:.2f})")

    axes[1, 1].imshow(phantom[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 1].imshow(lesion_mask[:, :, slice_idx].T, cmap='Reds', origin='lower', alpha=0.8)
    axes[1, 1].set_title(f"Lesion Segmentation (Dice={d_lesion:.2f})")

    axes[1, 2].imshow(phantom[:, :, slice_idx].T, cmap='gray', origin='lower', alpha=0.5)
    axes[1, 2].imshow(vessel_mask[:, :, slice_idx].T, cmap='Blues', origin='lower', alpha=0.8)
    axes[1, 2].set_title(f"Vessel Segmentation (Dice={d_vessel:.2f})")

    for ax in axes.flat:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    plt.suptitle("Ultrasound Preprocessing & Segmentation Pipeline", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/02_preprocessing_pipeline.png", dpi=150)
    print(f"Saved: {RESULTS_DIR}/02_preprocessing_pipeline.png")
