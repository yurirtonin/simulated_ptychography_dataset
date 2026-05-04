import numpy as np
import matplotlib.pyplot as plt

def estimate_array_size_gb(shape: tuple, dtype: np.dtype) -> float:
    """Estimates the size of a NumPy array in gigabytes given its shape and data type."""
    num_elements = np.prod(shape)  # Total number of elements
    element_size = np.dtype(dtype).itemsize  # Bytes per element
    return (num_elements * element_size) / (1024**3)  # Convert bytes to GB


def wavelength_meters_from_energy_keV(energy_keV):
    """
    Convert energy in keV to wavelength in meters.
    """
    speed_of_light = 299792458        # Speed of Light [m/s]
    planck         = 4.135667662E-18  # Plank constant [keV*s]
    return planck * speed_of_light / energy_keV


def pad_to_size(arr, N, mode='constant', constant_values=0):
    """
    Pad a 2D array to shape (N, N), centering the original array.
    """
    pad_y = (N - arr.shape[0]) // 2
    pad_x = (N - arr.shape[1]) // 2
    pad_width = ((pad_y, N - arr.shape[0] - pad_y), (pad_x, N - arr.shape[1] - pad_x))
    return np.pad(arr, pad_width, mode=mode, constant_values=constant_values)

def apply_vignette(image, method='tukey', **kwargs):
    """
    Apply a vignetting window to a 2D complex image.
    """
    from scipy.signal.windows import tukey, chebwin

    DC = np.mean(image)
    image_zero = image - DC
    rows, cols = image.shape
    if method.lower() == 'gaussian':
        sigma = kwargs.get('sigma', 5)
        x = np.linspace(- (cols - 1) / 2., (cols - 1) / 2., cols)
        y = np.linspace(- (rows - 1) / 2., (rows - 1) / 2., rows)
        win_x = np.exp(-0.5 * (x / sigma) ** 2)
        win_y = np.exp(-0.5 * (y / sigma) ** 2)
    elif method.lower() == 'hann':
        win_x = np.hanning(cols)
        win_y = np.hanning(rows)
    elif method.lower() == 'hamming':
        win_x = np.hamming(cols)
        win_y = np.hamming(rows)
    elif method.lower() == 'tukey':
        alpha = kwargs.get('alpha', 0.5)
        win_x = tukey(cols, alpha=alpha)
        win_y = tukey(rows, alpha=alpha)
    elif method.lower() == 'chebyshev':
        attenuation = kwargs.get('attenuation', 100)
        win_x = chebwin(cols, at=attenuation)
        win_y = chebwin(rows, at=attenuation)
    else:
        raise ValueError("Unsupported window method. Choose one of: gaussian, hann, hamming, tukey, or chebyshev")
    mask = np.outer(win_y, win_x)
    image_vignetted = image_zero * mask + DC
    return image_vignetted, mask

def convert_complex_to_RGB_HS(ComplexImg, mag_max=None):
    """
    Convert a complex image into an RGB image where:
      - Hue encodes the phase.
      - Value encodes the normalized amplitude so that low amplitudes are black.
      - Saturation is fixed to 1.
    
    Parameters:
      ComplexImg : 2D complex array.
      mag_max    : Maximum amplitude used for normalization. If None, uses the actual maximum.
    
    Returns:
      RGB : An RGB image (height x width x 3) as a NumPy array.
    """
    from matplotlib.colors import hsv_to_rgb

    # Compute amplitude and phase.
    Amps = np.abs(ComplexImg)
    Phases = np.angle(ComplexImg)
    
    # Normalize phase from [-π, π] to [0,1]
    Phases_norm = (Phases + np.pi) / (2 * np.pi)
    
    # Normalize amplitude for the value channel.
    if mag_max is None:
        mag_max = Amps.max()
    Amps_norm = np.clip(Amps / mag_max, 0, 1)
    
    # Build the HSV image:
    # - Hue from the normalized phase.
    # - Saturation fixed to 1.
    # - Value from the normalized amplitude (low amplitude → low value → black).
    HSV = np.zeros(ComplexImg.shape + (3,), dtype=np.float32)
    HSV[..., 0] = Phases_norm  # Hue
    HSV[..., 1] = 1            # Saturation
    HSV[..., 2] = Amps_norm    # Value
    
    # Convert HSV to RGB.
    RGB = hsv_to_rgb(HSV)
    return RGB


def get_RGB_probe(probe,max_magnitude=2):
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    RGB = convert_complex_to_RGB_HS(probe, mag_max=max_magnitude)
    # --- Create a colorbar for the phase ---
    # The phase was mapped from [-π, π] to [0, 1] for the hue.
    phase_norm = Normalize(vmin=-np.pi, vmax=np.pi)
    # Create a dummy ScalarMappable with the 'hsv' colormap.
    sm = cm.ScalarMappable(cmap='hsv', norm=phase_norm)
    return RGB, sm

def plot_amp_phase_complex(img, colormap='viridis', amp_vmin=None,amp_vmax=None,vmin=None, vmax=None):
    """
    Plots the magnitude, phase, and HSV representation of a 2D complex img.

    Parameters:
    - img: numpy.ndarray
        The 2D complex-valued img array. Expected shape: (height, width).
    - colormap: str, optional
        Colormap for the magnitude and phase plots. Default is 'viridis'.
    - vmin: float, optional
        Minimum value for the amplitude scaling in the HSV plot. 
        If None, it is calculated automatically based on the data.
    - vmax: float, optional
        Maximum value for the amplitude scaling in the HSV plot. 
        If None, it is calculated automatically based on the data.

    Raises:
    - ValueError: If the img array is not 2-dimensional.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    
    # Validate img dimensionality
    if img.ndim != 2:
        raise ValueError("img array must be 2-dimensional (height, width).")

    # Compute magnitude and statistics for automatic scaling
    img_mag = np.abs(img)
    mean_img_mag = np.mean(img_mag)
    std_img_mag = np.std(img_mag)

    if amp_vmin is None:
        automatic_vmin = img_mag.min()
    else:
        automatic_vmin = mean_img_mag - 3 * std_img_mag

    if amp_vmax is None:
        automatic_vmax = img_mag.max()
    else:
        automatic_vmax = mean_img_mag + 3 * std_img_mag

    # Handle user-specified vmin and vmax for HSV normalization
    if vmin is not None and vmax is not None:
        if vmin >= vmax:
            raise ValueError("vmin should be less than vmax.")
        amp_norm = (img_mag - vmin) / (vmax - vmin)
    elif vmin is not None:
        amp_norm = (img_mag - vmin) / (automatic_vmax - vmin)
    elif vmax is not None:
        amp_norm = (img_mag - automatic_vmin) / (vmax - automatic_vmin)
    else:
        amp_norm = (img_mag - automatic_vmin) / (automatic_vmax - automatic_vmin)
    
    amp_norm = np.clip(amp_norm, 0, 1)

    # Compute phase
    img_phase = np.angle(img)
    # Normalize phase to [0,1] for hue
    hue = (img_phase + np.pi) / (2 * np.pi)

    # Create HSV image: hue=phase, saturation=amplitude, value=amplitude
    hsv_image = np.stack([hue, amp_norm, amp_norm], axis=-1)
    rgb_image = hsv_to_rgb(hsv_image)

    # Create figure and grid layout
    fig = plt.figure(figsize=(18, 6))
    gs = plt.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    # Magnitude Plot
    ax_mag = fig.add_subplot(gs[0, 0])
    # Automatic scaling for magnitude plot
    vmin_mag = automatic_vmin
    vmax_mag = automatic_vmax
    im0 = ax_mag.imshow(img_mag, vmin=vmin_mag, vmax=vmax_mag, cmap=colormap)
    ax_mag.set_title('Magnitude')
    fig.colorbar(im0, ax=ax_mag, fraction=0.046, pad=0.04)

    # Phase Plot
    ax_phase = fig.add_subplot(gs[0, 1])
    im1 = ax_phase.imshow(img_phase, vmin=-np.pi, vmax=np.pi, cmap=colormap)
    ax_phase.set_title('Phase')
    fig.colorbar(im1, ax=ax_phase, fraction=0.046, pad=0.04)

    # HSV Plot
    ax_hsv = fig.add_subplot(gs[0, 2])
    im2 = ax_hsv.imshow(rgb_image)
    ax_hsv.set_title('Complex img')

    # Color Map Legend for HSV
    ax_cbar = fig.add_subplot(gs[0, 3])
    phase = np.linspace(0, 1, 256)  # normalized phase
    if vmin is not None or vmax is not None:
        # Adjust amplitude depending on provided vmin and vmax
        if vmin is not None and vmax is not None:
            amp_start = 0
            amp_end = 1
        elif vmin is not None:
            amp_start = 0
            amp_end = (automatic_vmax - vmin) / (automatic_vmax - automatic_vmin)
        else:  # vmax is not None
            amp_start = (vmax - automatic_vmin) / (automatic_vmax - automatic_vmin)
            amp_end = 1
        amplitude = np.linspace(amp_start, amp_end, 256)
    else:
        amplitude = np.linspace(0, 1, 256)  # normalized amplitude

    phase_grid, amplitude_grid = np.meshgrid(phase, amplitude)
    color_map = hsv_to_rgb(np.dstack((phase_grid, amplitude_grid, amplitude_grid)))
    ax_cbar.imshow(color_map, aspect='auto', origin='lower')
    ax_cbar.set_xticks([0, 255])
    ax_cbar.set_xticklabels(['-π', 'π'])
    ax_cbar.set_yticks([0, 255])
    ax_cbar.set_yticklabels(['0', 'Max'])
    ax_cbar.set_xlabel('Phase')
    ax_cbar.set_ylabel('Amplitude')

    plt.tight_layout()
    plt.show()


def calculate_photon_density_map(positions, beam, obj_shape, pixel_size,photon_numbers=None):
    """
    Calculate the photon density map and dose from the given inputs and plots relevant maps.

    Parameters:
      positions         : numpy.ndarray of shape (N,2)
                          Array of (y, x) positions in meters.
      beam              : numpy.ndarray
                          Complex array representing beam amplitude. The intensity is |beam|^2.
      photon_numbers    : numpy.ndarray or None
                          Array of photon numbers corresponding to each scan position.
                          If None, a mean value of 1.0 is assumed.
      obj_shape         : tuple
                          Shape (height, width) of the object/image area in pixels.
      pixel_size        : float
                          Pixel size in meters.
    
    Returns:
      dose              : float
                          Computed dose in J/kg.
      photon_density_map: numpy.ndarray
                          The computed photon density map (per pixel).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from skimage import data

    # Convert positions from meters to pixel indices
    positions_pixels = np.round(positions / pixel_size).astype(int)
    
    # Compute beam intensity at object: usually |beam|^2.
    beam_intensity_at_obj = np.abs(beam)**2

    # Normalize the photon numbers if available; otherwise default to 1.0 per position.
    if photon_numbers is not None:
        mean_photon_number = photon_numbers.mean()
        photon_numbers = photon_numbers / mean_photon_number  # Not necessarily same as Wilke 2012.
    else:
        mean_photon_number = 1.0
        photon_numbers = np.ones(positions_pixels.shape[0], dtype=np.float64)
    
    # Initialize the photon density map with zeros.
    photon_density_map = np.zeros(obj_shape, dtype=np.float64)
    
    # For each scan position add the corresponding photon intensity scaled by the photon number.
    beam_shape = beam_intensity_at_obj.shape
    for i, pos in enumerate(positions_pixels):
        y_start, x_start = pos[0], pos[1]
        # Adding the beam profile at the given pixel location.
        photon_density_map[y_start:y_start+beam_shape[0], x_start:x_start+beam_shape[1]] += \
            photon_numbers[i] * beam_intensity_at_obj

    return photon_density_map