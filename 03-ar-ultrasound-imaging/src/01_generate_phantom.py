"""
01_generate_phantom.py
----------------------
Génère un phantom abdominal 3D synthétique avec bruit speckle réaliste.
Un phantom est un objet simulé utilisé pour valider les algorithmes
d'imagerie sans données patient réelles.
"""
import os
import numpy as np
from scipy.ndimage import gaussian_filter

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def generate_speckle_noise(shape, mean=1.0, sigma=0.15):
    real = np.random.normal(0, sigma, shape)
    imag = np.random.normal(0, sigma, shape)
    speckle = np.sqrt(real**2 + imag**2)
    return speckle / speckle.mean() * mean


def generate_phantom_3d(size=128, seed=None):
    if seed is not None:
        np.random.seed(seed)

    vol = np.zeros((size, size, size), dtype=np.float32)
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    wall_mask = (Z > 0.7) & (Z < 0.85)
    vol[wall_mask] = 0.6

    liver_mask = ((X/0.7)**2 + (Y/0.6)**2 + ((Z-0.1)/0.5)**2) < 1.0
    liver_mask &= (Z < 0.7)
    vol[liver_mask] = 0.45

    vessel_mask = ((X - 0.2)**2 + (Y + 0.1)**2 < 0.08**2) & (Z > -0.3) & (Z < 0.4)
    vol[vessel_mask] = 0.15

    lesion_mask = ((X + 0.25)**2 + (Y - 0.15)**2 + (Z + 0.1)**2) < 0.12**2
    vol[lesion_mask] = 0.75

    noise_texture = gaussian_filter(np.random.randn(size, size, size), sigma=2)
    vol[liver_mask] += noise_texture[liver_mask] * 0.05

    speckle = generate_speckle_noise(vol.shape, mean=1.0, sigma=0.2)
    vol = vol * speckle
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

    # NOTE: vessel_mask est defini comme sous-ensemble geometrique de liver_mask
    # dans les donnees (le cylindre est bien a l'interieur de l'ellipsoide),
    # on le garde independant pour l'evaluation.
    gt_masks = {
        "wall": wall_mask,
        "liver": liver_mask,
        "vessel": vessel_mask,
        "lesion": lesion_mask,
    }
    return vol, gt_masks


if __name__ == "__main__":
    print("Generating 3D ultrasound phantom...")
    phantom, gt_masks = generate_phantom_3d(size=128, seed=42)

    np.save(f"{DATA_DIR}/phantom_us_3d.npy", phantom)
    for name, mask in gt_masks.items():
        np.save(f"{DATA_DIR}/gt_{name}_mask.npy", mask)

    print(f"Saved: {DATA_DIR}/phantom_us_3d.npy")
    print(f"  Shape: {phantom.shape}")
    print(f"  Range: [{phantom.min():.3f}, {phantom.max():.3f}]")
    print("Saved ground-truth masks: wall, liver, vessel, lesion")