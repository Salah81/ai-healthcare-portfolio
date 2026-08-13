"""
03_visualize_3d_vtk.py
----------------------
Visualisation 3D du volume échographique avec PyVista (VTK).
Affiche:
  - Le volume abdominal en volume rendering
  - Un plan de coupe cyan simulant la sonde US
  - Un repère 3D (axes)
  
Usage:
  pip install pyvista -q
  python src/03_visualize_3d_vtk.py
"""
import os
import numpy as np
import pyvista as pv

DATA_DIR = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def mask_to_mesh(mask, spacing=(1, 1, 1)):
    """Convertit un masque binaire 3D en surface (marching cubes) via VTK/PyVista."""
    grid = pv.ImageData()
    grid.dimensions = np.array(mask.shape) + 1  # +1: valeurs definies aux noeuds, pas aux cellules
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.cell_data["mask"] = mask.flatten(order="F").astype(float)
    grid = grid.cells_to_points("mask")
    surface = grid.contour(isosurfaces=[0.5], scalars="mask")
    return surface.smooth(n_iter=30, relaxation_factor=0.1)


def main():
    print("Loading segmented masks (from 02_preprocess_segment.py)...")
    liver_mask = np.load(f"{DATA_DIR}/liver_mask.npy")
    vessel_mask = np.load(f"{DATA_DIR}/vessel_mask.npy")
    lesion_mask = np.load(f"{DATA_DIR}/lesion_mask.npy")

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.set_background("#141428")

    # Foie: enveloppe translucide (contexte anatomique)
    liver_surf = mask_to_mesh(liver_mask)
    plotter.add_mesh(liver_surf, color="#5da87a", opacity=0.25, smooth_shading=True, name="liver")

    # Vaisseau: structure tubulaire, bien visible
    vessel_surf = mask_to_mesh(vessel_mask)
    plotter.add_mesh(vessel_surf, color="#3f7fd6", opacity=0.9, smooth_shading=True, name="vessel")

    # Lesion: nodule, couleur d'alerte
    lesion_surf = mask_to_mesh(lesion_mask)
    plotter.add_mesh(lesion_surf, color="#d64545", opacity=0.95, smooth_shading=True, name="lesion")

    # Plan de coupe simulant la sonde echographique (inchange)
    probe = pv.Plane(
        center=(64, 64, 100),
        direction=(0.3, 0.1, 1.0),
        i_size=90,
        j_size=70,
    )
    plotter.add_mesh(probe, color="cyan", opacity=0.25, show_edges=True, line_width=2, name="probe_plane")

    plotter.add_axes(line_width=3, color="white")

    # Camera: vraie vue 3/4, cadree sur le volume entier (0-128 dans chaque axe)
    plotter.camera_position = [
        (280, -180, 220),   # position (vue oblique, reculee pour cadrer tout le volume)
        (64, 64, 64),       # point focal: centre du volume
        (0, 0, 1),          # up vector
    ]
    plotter.camera.zoom(1.15)

    output_path = f"{RESULTS_DIR}/03_vtk_volume_rendering.png"
    plotter.show(screenshot=output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()