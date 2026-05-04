# simulated_ptychography_dataset

A Python package for generating simulated datasets for near- and far-field ptychography experiments. It provides tools to create realistic test phantoms, Gaussian beam probes, scan grids (raster, jittered, Fermat spiral), and forward-simulate ptychographic diffraction patterns with optional noise and detector masking.

---

## Package structure

```
simulated_ptychography_dataset/
├── __init__.py          # Re-exports all public symbols
├── phantoms.py          # Phantom class – complex test objects
├── misc.py              # Utility functions (probes, visualisation, photon density)
├── ptychography.py      # Core functions and PtychographyDataset class
├── example.ipynb        # End-to-end usage example
├── pyproject.toml
└── setup.py
```

### Key modules

| Module | Contents |
|---|---|
| `phantoms` | `Phantom` – load a complex object from built-in datasets (`dicty`, `dicty_multi`, …) with optional beta/delta values from `xraydb` |
| `misc` | `gaussian_beam`, `pad_to_size`, `apply_vignette`, `get_RGB_probe`, `plot_amp_phase_complex`, `calculate_photon_density_map`, `wavelength_meters_from_energy_keV` |
| `ptychography` | `get_scan_positions`, `calculate_object_shape`, `convert_probe_positions_to_pixel_units`, `get_farfield_pixel_size`, `PtychographyDataset` |

---

## Installation

Requires Python ≥ 3.9. Clone the repository and install in editable mode:

```bash
git clone <repo-url> simulated_ptychography_dataset
cd simulated_ptychography_dataset
pip install -e .
```

### Dependencies

Installed automatically via `pip`:

- `numpy`, `matplotlib`, `scipy`, `scikit-image`, `h5py`, `tqdm`
- `xraydb` – X-ray optical constants (for `Phantom.get_complex_with_betadelta`)
- `hotopy` – Fresnel propagators and holographic datasets (must be available in your environment)

---

## Quick start

Open and run `example.ipynb`, or use the package programmatically:

```python
import numpy as np
from simulated_ptychography_dataset import (
    Phantom,
    gaussian_beam,
    get_scan_positions,
    calculate_object_shape,
    calculate_photon_density_map,
    plot_amp_phase_complex,
)

# --- Experiment parameters ---
E_keV        = 8.0
detector_pixel = 75e-6          # Eiger pixel size [m]
n_pixels     = 1024             # detector size
z1           = 3e-3             # source-to-sample distance [m]
z2           = 5.097            # sample-to-detector distance [m]
wavelength   = 1.23984 / E_keV * 1e-9
obj_pixel    = wavelength * z2 / (n_pixels * detector_pixel)

# --- Phantom (complex object) ---
phantom = Phantom(source='dicty')
complex_obj, beta, delta, _ = phantom.get_complex_with_betadelta(energy_keV=E_keV)

# --- Probe ---
w0    = 60e-9   # beam waist [m]
probe = gaussian_beam(n_pixels, obj_pixel, z1, w0, wavelength)
probe /= np.abs(probe).max()

# --- Scan grid ---
# Returns three (N, 3) arrays in [z, y, x] order (metres for 'positions',
# pixels for 'positions_pxls' and 'positions_pxls_rounded')
positions, positions_pxls, positions_pxls_rounded = get_scan_positions(
    xmin=-5, xmax=5,        # scan range [µm]
    ymin=-5, ymax=5,
    n_points=10,
    grid_type='jitter',     # 'raster' | 'raster_noise'/'jitter' | 'fermat'
    z1=z1,
    obj_pixel=obj_pixel,
    plot=True,
)

# --- Object array sized to fit all scan positions ---
obj_shape = calculate_object_shape(positions_pxls_rounded, probe)

# --- Photon density map ---
# positions[:, 1:3] extracts the (y, x) columns in metres
photon_map = calculate_photon_density_map(positions[:, 1:3], probe, obj_shape, obj_pixel)
```

### Simulating diffraction patterns

```python
from hotopy.holo.propagation import FresnelTFPropagator

magnification      = (z1 + z2) / z1
magnified_pixel    = detector_pixel / magnification
magnified_distance = z2 / magnification
fresnel_number     = obj_pixel**2 / wavelength / magnified_distance

propagator = FresnelTFPropagator(probe.shape, fresnel_number)

ny, nx = probe.shape
ptychogram = np.empty((len(positions_pxls_rounded), ny, nx), dtype=np.float32)

for i, (_, posy, posx) in enumerate(positions_pxls_rounded):
    posy, posx   = int(posy), int(posx)
    exit_wave    = probe * obj[posy:posy+ny, posx:posx+nx]
    ptychogram[i] = np.abs(np.fft.fftshift(np.fft.fft2(exit_wave)))**2
```

---

## Scan grid types

| `grid_type` | Description |
|---|---|
| `'raster'` | Regular rectangular grid with `n_points × n_points` positions |
| `'raster_noise'` or `'jitter'` | Raster grid with small random offsets (Gaussian noise, σ = 0.1 µm) |
| `'fermat'` | Fermat spiral – quasi-random coverage, avoids periodicity artefacts |

---

## Position array convention

All position arrays are shaped `(N, 3)` with columns `[z, y, x]`:

- **`positions`** – metre-space coordinates snapped to the nearest pixel grid point
- **`positions_pxls`** – continuous pixel-space coordinates (float)
- **`positions_pxls_rounded`** – integer-rounded pixel-space coordinates

Pass `positions[:, 1:3]` (the `[y, x]` slice) to `calculate_photon_density_map`.

