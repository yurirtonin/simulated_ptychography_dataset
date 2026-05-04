import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tqdm, os, sys

from hotopy.holo.propagation import FresnelTFPropagator

from .phantoms import Phantom
from .misc import wavelength_meters_from_energy_keV, plot_amp_phase_complex, get_RGB_probe, apply_vignette, pad_to_size


#TODO: implement different probe types
#TODO: implement different object types

##### FUNCTIONS

def check_for_nans(data,i):
    """
    Check if the data contains NaN values.
    """
    if np.any(np.isnan(data)):
        print("Nans found",i)
    return True

def get_farfield_pixel_size(wavelength,detector_distance, detector_pixel_size,n_of_pixels):
    return wavelength * detector_distance / (detector_pixel_size * n_of_pixels)


def check_data_shapes(probe,obj,data,mask,obj_support,source_support):
    """
    Check the shapes of the data, probe, object, mask, and supports.
    """

    if data is None:
        data = np.squeeze(data)
        if len(data.shape) > 3:
            raise ValueError(f"Data shape {data.shape} is not 2D or 3D.")
    if probe is not None:
        if probe.shape[-2] != data.shape[-2] or probe.shape[-1] != data.shape[-1]:
            raise ValueError(f"Probe shape {probe.shape} does not match data shape {data.shape}.")
    if mask is not None:
        if mask.shape[-2] != data.shape[-2] or mask.shape[-1] != data.shape[-1]:
            raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}.")
    if obj is not None:
        if obj.shape[-2] < probe.shape[-2] or obj.shape[-1] < probe.shape[-1]:
            raise ValueError(f"Object shape {obj.shape} is smaller than probe shape {probe.shape}.")
    if obj_support is not None:
        if obj_support.shape[-2] != obj.shape[-2] or obj_support.shape[-1] != obj.shape[-1]:
            raise ValueError(f"Object support shape {obj_support.shape} does not match object shape {obj.shape}.")
    if source_support is not None:
        if source_support.shape[-2] != probe.shape[-2] or source_support.shape[-1] != probe.shape[-1]:
            raise ValueError(f"Source support shape {source_support.shape} does not match probe shape {probe.shape}.")

def check_data_types(probe,obj,data,mask,obj_support,source_support):
    """
    Check the types of the data, probe, object, mask, and supports.
    """
    if probe is not None and probe.dtype != np.complex64:
        print(f"Probe dtype {probe.dtype} is not complex64. Converting to complex64.")
        probe = probe.astype(np.complex64)
    if obj is not None and obj.dtype != np.complex64:
        print(f"Object dtype {obj.dtype} is not complex64. Converting to complex64.")
        obj = obj.astype(np.complex64)
    if data is not None and data.dtype != np.float32:
        print(f"Data dtype {data.dtype} is not float32. Converting to float32.")
        data = data.astype(np.float32)

def convert_probe_positions_to_pixel_units(positions, pixel_size, make_positive=True, round_positions=True, offset=0):
    # Convert positions to pixel units
    positions_pxls = positions.copy()
    positions_pxls[:,1:3] = positions[:,1:3] / pixel_size

    if make_positive:
        positions_pxls[:,-2] -= positions_pxls[:,-2].min()
        positions_pxls[:,-1] -= positions_pxls[:,-1].min()

    if round_positions:
        positions_pxls_rounded = positions_pxls.copy()
        positions_pxls_rounded[:,1:3] = np.round(positions_pxls[:,1:3]).astype(int)

    if offset > 0:
        positions_pxls[:,1:3] += offset
        positions_pxls_rounded[:,1:3] += offset

    return positions_pxls, positions_pxls_rounded

def calculate_object_shape(positions,probe):
    sizey = positions[:,-2].max()+probe.shape[-2]
    sizex = positions[:,-1].max()+probe.shape[-1]

    # # check if sizey and sizex are integers
    # if not (float(sizey).is_integer() and float(sizex).is_integer()):
    #     raise ValueError(f"Calculated object shape ({sizey}, {sizex}) is not integer. Check probe size and positions.")
    # else: 
        # sizey = int(sizey)
        # sizex = int(sizex)
    sizey = int(sizey)
    sizex = int(sizex)

    object_shape = (sizey,sizex)
    return object_shape


def gaussian_beam(array_size, pixel_size, z, w0, wavelength):
    """
    Calculate the complex electric field of a Gaussian beam at position z on a square grid.

    Arguments:
      array_size : int, number of pixels along one axis (assumed square grid)
      pixel_size : float, size of each pixel [m]
      z          : propagation distance along the beam axis [m]
      w0         : beam waist at focus (z=0) [m]
      wavelength : wavelength of the beam [m]

    Returns:
      E          : Complex electric field of the Gaussian beam at each grid point, shape (array_size, array_size)
    """
    # Create centered coordinate grid
    n = array_size // 2
    if array_size % 2 == 0:
        coords = np.arange(-n, n) * pixel_size
    else:
        coords = np.arange(-n, n + 1) * pixel_size
    X, Y = np.meshgrid(coords, coords, indexing='xy')

    k = 2 * np.pi / wavelength            # wave number
    z_R = np.pi * w0**2 / wavelength      # Rayleigh range
    r = np.sqrt(X**2 + Y**2)              # radial coordinate

    # Beam radius as a function of z
    wz = w0 * np.sqrt(1 + (z/z_R)**2)

    # Radius of curvature—handle the z == 0 case to avoid division by zero.
    if z == 0:
        Rz = np.inf
    else:
        Rz = z * (1 + (z_R/z)**2)

    # Gouy phase
    psi = np.arctan(z / z_R)

    # Field amplitude prefactor ensuring the correct normalization (if desired)
    amplitude = w0 / wz

    # Calculate the quadratic phase term. When Rz is infinite the term vanishes.
    # curvature_phase = 1 if np.isinf(Rz) else np.exp(-1j * k * r**2 / (2 * Rz))

    # Complex field of the Gaussian beam
    field = amplitude * np.exp(-r**2 / wz**2) * (1 if np.isinf(Rz) else np.exp(1j * k * r**2 / (2 * Rz))) * np.exp(1j * k * z) * np.exp(-1j * psi) # sign convention adjusted for divergent beam. See Jacobsen's book
    
    return field

def get_scan_positions(xmin, xmax, ymin, ymax, n_points, grid_type, z1, obj_pixel, fermat_c=0.565, plot=False):
    """
    Standalone module-level function to generate scan positions.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Scan range in microns.
    n_points : int
        Number of points per axis for raster grids.
    grid_type : str
        One of 'raster', 'raster_noise' / 'jitter', 'fermat'.
    z1 : float
        Source-to-object distance in meters. Prepended as the z column.
    obj_pixel : float
        Object-plane pixel size in meters. Used to convert positions to pixel units.
    fermat_c : float, optional
        Scaling constant for the Fermat spiral (default 0.565).
    plot : bool, optional
        If True, plot the generated scan grids.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Rounded positions in meters [z, y, x] after snapping to pixel grid.
    positions_pxls : np.ndarray, shape (N, 3)
        Positions in pixel units [z, y, x] (float).
    positions_pxls_rounded : np.ndarray, shape (N, 3)
        Positions in pixel units [z, y, x] (integer-rounded).
    """
    # --- build raster grid ---
    x_r = np.linspace(xmin, xmax, n_points)
    y_r = np.linspace(ymin, ymax, n_points)
    Xr, Yr = np.meshgrid(x_r, y_r)
    raster_points = np.column_stack((Xr.ravel(), Yr.ravel()))

    # --- raster with noise ---
    noise = np.random.normal(0, 0.1, raster_points.shape)
    raster_noise_points = raster_points + noise

    # --- Fermat spiral ---
    n_target = n_points ** 2
    N_spiral = int(np.ceil((np.pi / 2) * n_target))
    golden_angle = np.deg2rad(137.508)
    idx = np.arange(N_spiral)
    r_f = fermat_c * np.sqrt(idx)
    theta_f = idx * golden_angle
    fermat_spiral = np.column_stack((r_f * np.cos(theta_f), r_f * np.sin(theta_f)))
    outside = ((fermat_spiral[:, 0] < xmin) | (fermat_spiral[:, 0] > xmax) |
               (fermat_spiral[:, 1] < ymin) | (fermat_spiral[:, 1] > ymax))
    fermat_spiral = fermat_spiral[~outside]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(22, 6))
        for ax_, pts, title, color in zip(
            axs,
            [raster_points, raster_noise_points, fermat_spiral],
            ["Regular Raster", "Raster with Noise", "Fermat Spiral"],
            ['blue', 'red', 'green'],
        ):
            ax_.plot(pts[:, 0], pts[:, 1], c=color, marker='o', markersize=5, linestyle='-')
            ax_.set_title(title)
            ax_.set_xlabel("x (µm)")
            ax_.set_ylabel("y (µm)")
            ax_.axis("equal")
            ax_.grid(True)
        plt.tight_layout()
        plt.show()

    # --- select grid and convert µm → metres ---
    if grid_type == 'raster':
        pts_m = raster_points * 1e-6
    elif grid_type in ('raster_noise', 'jitter'):
        pts_m = raster_noise_points * 1e-6
    elif grid_type == 'fermat':
        pts_m = fermat_spiral * 1e-6
    else:
        raise ValueError("grid_type must be one of 'raster', 'raster_noise' (or 'jitter'), 'fermat'")

    # roll so columns become [y, x] then prepend z column → [z, y, x]
    pts_m = np.roll(pts_m, shift=1, axis=1)
    z_col = np.full((pts_m.shape[0], 1), z1)
    positions_m = np.hstack((z_col, pts_m))  # (N, 3)  [z, y, x]

    positions_pxls, positions_pxls_rounded = convert_probe_positions_to_pixel_units(positions_m, obj_pixel)

    # snap positions in metres to the pixel grid to avoid sub-pixel rounding errors
    positions = positions_pxls_rounded * obj_pixel

    return positions, positions_pxls, positions_pxls_rounded


################### CLASSES
    
class PtychoArrays: # NEEDS TESTING 
    """
    Class to carry probe, object, data and supporting arrays, like mask and support.
    The reading strategy is as follows:
    - If the entry is a string:
        - If `self.path` is not an empty string, prioritize loading from the HDF5 file at `self.path`.
        - If it ends with '.npy', load it as a NumPy array.
        - If it ends with '.npz', load the specific dataset from the file.
        - If it ends with '.h5' or '.hdf5', load the dataset from the HDF5 file.
        - If it is a dataset path in an HDF5 file, load it from the file specified by `self.path`.
    - If the entry is a NumPy array, use it directly.
    - If the entry is None, return None.
    - Raise appropriate errors for unsupported file extensions or invalid types.
    """
    def __init__(self, probe=None, obj=None, data=None, mask=None, obj_support=None, source_support=None, path=''):
        self.path = path
        self.probe = self._load_data(probe, 'probe')
        self.obj = self._load_data(obj, 'obj')
        self.data = self._load_data(data, 'data')
        self.mask = self._load_data(mask, 'mask')
        self.obj_support = self._load_data(obj_support, 'obj_support')
        self.source_support = self._load_data(source_support, 'source_support')

    def _load_data(self, entry, name):
        """
        Check the type of the entry. Load the data based on the file extension or HDF5 dataset.
        Prioritize loading from `self.path` if it is not empty.
        Raise an error for invalid types.
        """
        # If entry is None, return None (will be generated later if needed)
        if entry is None:
            return None

        print("Warning: reading data from file is not fully tested yet. Proceed with caution.")

        # If entry is a numpy array, use it directly
        if isinstance(entry, np.ndarray):
            return entry
        
        # If entry is a string, it's a file path or dataset name
        if isinstance(entry, str):
            # If self.path is set and entry has no file extension, treat it as a dataset path in self.path
            if self.path and '.' not in entry:
                with h5py.File(self.path, 'r') as f:
                    if entry in f:
                        return f[entry][:]
                    else:
                        raise ValueError(f"Dataset '{entry}' not found in HDF5 file: {self.path}")
            
            # Otherwise, load based on file extension
            if entry.endswith('.npy'):
                return np.load(entry)
            elif entry.endswith('.npz'):
                data = np.load(entry)
                if name in data:
                    return data[name]
                else:
                    raise ValueError(f"Dataset '{name}' not found in .npz file: {entry}")
            elif entry.endswith('.h5') or entry.endswith('.hdf5'):
                with h5py.File(entry, 'r') as f:
                    if name in f:
                        return f[name][:]
                    else:
                        raise ValueError(f"Dataset '{name}' not found in HDF5 file: {entry}")
            else:
                raise ValueError(f"Unsupported file extension for {name}: {entry}")
        
        # If we get here, the type is invalid
        raise TypeError(f"Invalid type for {name}: {type(entry)}. Must be None, str, or numpy.ndarray.")



class Inputs:
    def __repr__(self):
        return str(self.__dict__)
    pass

class FraunhoferPropagator:
    """this propagator does not include the proper phase factors, so just the intensities should be used"""
    def __call__(self, u0):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u0), norm="ortho"))

    def inverse(self, u0):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u0), norm="ortho"))

class PtychographyDataset(PtychoArrays):
    """
    Class for creating a ptychography dataset
    """

    def __init__(self, probe=None, obj=None, data=None, mask=None, obj_support=None, source_support=None, path=''):
        super().__init__(probe, obj, data, mask, obj_support, source_support, path)
        self.initialize_default_attributes()

    def __call__(self):
        """
        Create synthetic ptychography dataset
        """
        # Calculate necessary metadata
        self.wavelength = wavelength_meters_from_energy_keV(self.energy)  # Convert keV to meters
        self.k_wavector = 2 * np.pi / self.wavelength

        self.check_and_set_default_values()  # Check and set default values for attributes
        
        self.figure_out_experiment_type()

        self.phantom, beta, delta, betadelta_ratio = Phantom(source=self.obj_type).get_complex_with_betadelta(element=self.obj_element, energy_keV=self.energy, plot=self.plot)
        
        if self.plot:
            plot_amp_phase_complex(self.phantom, colormap='gray')

        self.probe = self.get_probe()

        self.mask = self.get_mask()

        # Generate scan positions
        self.positions = self.get_scan_positions_for_this_experiment()
        
        # Get object
        self.obj = self.get_object()

        if self.obj.shape[-2] < self.probe.shape[-2] or self.obj.shape[-1] < self.probe.shape[-1]:
            raise ValueError(f"Object shape {self.obj.shape} is smaller than probe shape {self.probe.shape}. Use a smaller probe or larger object.")

        # Get diffraction data
        self.data = self.get_diffraction_data()

        # Check data shapes and types
        # check_data_shapes(self.probe, self.obj, self.data, self.mask, self.obj_support, self.source_support)
        # check_data_types(self.probe, self.obj, self.data, self.mask, self.obj_support, self.source_support)

    def initialize_default_attributes(self):
        """
        Initialize default attributes for the class.
        """
        self.plot = None
        self.energy = None
        self.detector_pixel_size = None
        self.z_values = None
        self.n_detector_pixels = None
        self.geometry = None
        self.obj_type = None
        self.obj_element = None
        self.obj_support = None
        self.source_plane_support = None
        self.probe_type = None
        self.probe_gaussian_waist = None
        self.probe_diameter = None
        self.probe_source_distance = None
        self.wavelength = None
        self.k_wavector = None
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.n_points = None
        self.grid_type = None

    def check_and_set_default_values(self):
        """
        Ensure all required attributes exist and set defaults if not.
        """
        if self.energy is None:
            self.energy = 8.0  # Default energy in keV
            print(f"Energy not set. Using default value: {self.energy} keV")
        elif self.energy < 0:
            self.energy = np.abs(self.energy)  # Ensure energy is positive
            print(f"WARNING: Energy provided is negative. Assuming absolute value: {self.energy} keV")

        if self.detector_pixel_size is None or self.detector_pixel_size <= 0:
            self.detector_pixel_size = 75e-6  # Default detector pixel size in meters
            print(f"Detector pixel size not set or invalid. Using default value: {self.detector_pixel_size} m")

        if self.z_values is None:
            self.z_values = [1e-3, 5e-3]  # Default z_values in meters
            print(f"z_values not set. Using default values: {self.z_values}")

        if self.n_detector_pixels is None:
            self.n_detector_pixels = 512  # Default number of detector pixels
            print(f"Number of detector pixels not set. Using default value: {self.n_detector_pixels}")

        if self.geometry is None:
            self.geometry = 'parallel'  # Default geometry
            print(f"Geometry not set. Using default value: {self.geometry}")

        if self.obj_type is None:
            self.obj_type = 'dicty'  # Default object type
            print(f"Object type not set. Using default value: {self.obj_type}")

        if self.obj_element is None:
            self.obj_element = 'H50C40N9O10S1'  # Default object element composition
            print(f"Object element not set. Using default value: {self.obj_element}")

        if self.probe_type is None:
            self.probe_type = 'gaussian'  # Default probe type
            print(f"Probe type not set. Using default value: {self.probe_type}")

        if self.probe_gaussian_waist is None:
            self.probe_gaussian_waist = 30e-9  # Default Gaussian waist in meters
            print(f"Probe Gaussian waist not set. Using default value: {self.probe_gaussian_waist} m")

        if self.probe_source_distance is None:
            self.probe_source_distance = 0.1  # Default source distance in meters
            print(f"Probe source distance not set. Using default value: {self.probe_source_distance} m")

    def figure_out_experiment_type(self):
        """
        Process and validate z_values and geometry
        """
        if np.isscalar(self.z_values):  # If numerical value is passed, convert to array
            self.z_values = np.array([float(self.z_values)])

        self.z_values = np.asarray(self.z_values, dtype=float)

        if not np.all(np.diff(self.z_values) > 0):
            raise ValueError("z_values must be in increasing order.") # check if self.z_values are in increasing order

        if self.geometry == 'parallel':
            if self.z_values.size == 1: # STANDARD FARFIELD PTYCHOGRAPHY
                self.experiment_type = 'PFF'
                print("PFF experiment type being used. Using Fraunhofer propagator.")
                self.detector_distance = self.z_values[0]
                self.obj_pixel = self.wavelength * self.detector_distance / (self.detector_pixel_size * self.n_detector_pixels)
                print(f"Object pixel size: {self.obj_pixel*1e9:.2f} nm")
                self.propagator = FraunhoferPropagator()
                

            elif self.z_values.size == 2: # STANDARD CONE BEAM PTYCHOGAPHY
                self.experiment_type = 'PNF'
                print("PNF experiment type being used. Using Fresnel propagator.")
                self.source_distance = self.z_values[0]
                self.detector_distance = self.z_values[1] - self.z_values[0]
                self.magnification = self.z_values[1] / self.z_values[0]
                self.obj_pixel = self.pixel_size/self.magnification
                self.detector_distance = self.detector_distance/self.magnification
                self.fresnel_number = self.obj_pixel**2/self.wavelength/self.detector_distance
                self.propagator = FresnelTFPropagator((self.n_detector_pixels,self.n_detector_pixels),self.fresnel_number)

        elif self.geometry == 'cone':
            
            if self.z_values.size == 2: # CONE BEAM PTYCHOGRAPHY WITH FARFIELD PROPAGATOR
                self.experiment_type = 'CFF'
                print("CFF experiment type being used. Using Fraunhofer propagator.")
                self.source_distance = self.z_values[0]
                self.detector_distance = self.z_values[1] - self.z_values[0]
                self.obj_pixel = self.wavelength * self.detector_distance / (self.detector_pixel_size * self.n_detector_pixels)
                self.propagator = FraunhoferPropagator()

            elif self.z_values.size >= 3:
                self.experiment_type = 'CMD' # multi-distance cone beam ptychography
                print("CMD experiment type being used.")
                print("WARNING: multi-distance scenario being used.")
                self.source_distance = self.z_values[0]
                self.detector_position = self.z_values[-1]
                self.magnifications = [self.z_values[-1] / zj for zj in self.z_values[:-1]]
                self.obj_pixel = self.pixel_size/self.magnifications[0]
                self.detector_distances = [(self.detector_position - zj)/mag for zj,mag in zip(self.z_values[:-1], self.magnifications)]
                self.fresnel_numbers = [ (self.wavelength * d) / (self.detector_pixel_size * self.n_detector_pixels)  for d in self.detector_distances]
                # propagators in this case are defined later during the reconstruction
        else:
            raise ValueError("Unknown geometry. Use 'parallel' or 'cone'.")

    def get_scan_positions_for_this_experiment(self):
        print(self.experiment_type)
        if self.experiment_type == "PFF" or self.experiment_type == "PNF" or self.experiment_type == "CFF":
            self.positions = self.get_scan_positions(self.xmin, self.xmax, self.ymin, self.ymax, self.n_points, self.grid_type, plot=self.plot)
        elif self.experiment_type == "CMD":
            self.positions = self.get_scan_positions(self.xmin, self.xmax, self.ymin, self.ymax, self.n_points, self.grid_type, plot=self.plot)
        else:
            raise ValueError(f"Experiment type {self.experiment_type} not recognized.")
        
        return self.positions

    def get_object(self):
        complex_obj = self.phantom.copy()[0::2, 0::2]  # downsample by factor of 2
        print(f"Probe shape: {self.probe.shape}. Object shape: {complex_obj.shape}")

        # complex_obj = match_obj_to_probe_shape(complex_obj, probe) # adjust obj size to match that of the probe
        # complex_obj = complex_obj[0:probe.shape[0], 0:probe.shape[1]] # crop object to match probe size
        complex_obj = pad_to_size(complex_obj, self.probe.shape[0]) # pad object to match probe size
        print(f"New object shape: {complex_obj.shape}")


        # Get new object array size given all scan positions
        obj_shape = calculate_object_shape(self.positions_pxls_rounded, self.probe)

        # Place complex_obj in the center of obj
        print(obj_shape, complex_obj.shape)
        start_y = (obj_shape[0] - complex_obj.shape[0]) // 2
        start_x = (obj_shape[1] - complex_obj.shape[1]) // 2
        obj = np.ones(obj_shape, dtype=np.complex64)
        obj[start_y:start_y + complex_obj.shape[0], start_x:start_x + complex_obj.shape[1]] = complex_obj

        print(f"Whole object shape: {obj.shape}", f"Probe shape: {self.probe.shape}", f"Positions shape: {self.positions_pxls_rounded.shape}", f'Mask shape: {self.mask.shape}')

        # photon_density_map = calculate_photon_density_map(self.positions_pxls_rounded, self.probe, obj_shape, self.obj_pixel)

        positions_xy = np.column_stack((self.positions_pxls_rounded[:, -1], self.positions_pxls_rounded[:, -2]))
        positions_xy += self.probe.shape[0]//2
        # Conversion functions: pixels <-> microns
        # (Assuming obj_pixel is defined in meters.)
        pix_to_micron = lambda x: x * self.obj_pixel * 1e6      # Convert pixel coordinate to microns.
        micron_to_pix = lambda x: x / (self.obj_pixel * 1e6)      # Convert microns back to pixels.

        if self.plot:
            # Create a 2x2 grid with constrained layout
            fig, ax = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

            #---------------------
            # Top left: Amplitude
            #---------------------
            im0 = ax[0, 0].imshow(np.abs(obj), cmap='viridis')
            fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.1)
            ax[0, 0].set_title('Amplitude')
            ax[0, 0].scatter(positions_xy[:, -1], positions_xy[:, -2], facecolors='none', edgecolors='red', marker='.', s=10)
            # Add secondary axes for Amplitude subplot
            secax_x00 = ax[0, 0].secondary_xaxis('top', functions=(pix_to_micron, micron_to_pix))
            secax_x00.set_xlabel('x (µm)')
            secax_y00 = ax[0, 0].secondary_yaxis('right', functions=(pix_to_micron, micron_to_pix))
            secax_y00.set_ylabel('y (µm)')

            #--------------------
            # Top right: Phase
            #--------------------
            im1 = ax[0, 1].imshow(np.angle(obj), cmap='viridis')
            fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.1)
            ax[0, 1].set_title('Phase')
            ax[0, 1].scatter(positions_xy[:, -1], positions_xy[:, -2],  facecolors='none', edgecolors='red', marker='.', s=10,label='Probe Array Center')
            ax[0,1].legend()
            # Add secondary axes for Phase subplot
            secax_x01 = ax[0, 1].secondary_xaxis('top', functions=(pix_to_micron, micron_to_pix))
            secax_x01.set_xlabel('x (µm)')
            secax_y01 = ax[0, 1].secondary_yaxis('right', functions=(pix_to_micron, micron_to_pix))
            secax_y01.set_ylabel('y (µm)')

            #--------------------
            # Bottom left: Probe
            #--------------------
            RGB, sm = get_RGB_probe(self.probe, max_magnitude=np.abs(self.probe).max())
            im2 = ax[1, 0].imshow(RGB)
            # Use fig.colorbar directly with ax=ax[1,0] for consistent placement
            cbar2 = fig.colorbar(sm, ax=ax[1, 0], fraction=0.046, pad=0.1)
            cbar2.set_label("Phase (radians)")
            ax[1, 0].set_title('Probe')
            # Add secondary axes for Probe subplot
            secax_x10 = ax[1, 0].secondary_xaxis('top', functions=(pix_to_micron, micron_to_pix))
            secax_x10.set_xlabel('x (µm)')
            secax_y10 = ax[1, 0].secondary_yaxis('right', functions=(pix_to_micron, micron_to_pix))
            secax_y10.set_ylabel('y (µm)')

            #------------------------------
            # Bottom right: Photon Density Map
            #------------------------------
            # im3 = ax[1, 1].imshow(photon_density_map, cmap='gray')
            ax[1, 1].set_title('Photon Density Map. Needs debugging')
            # fig.colorbar(im3, ax=ax[1, 1], fraction=0.046, pad=0.1)
            # # Add secondary axes for Photon Density Map subplot
            # secax_x11 = ax[1, 1].secondary_xaxis('top', functions=(pix_to_micron, micron_to_pix))
            # secax_x11.set_xlabel('x (µm)')
            # secax_y11 = ax[1, 1].secondary_yaxis('right', functions=(pix_to_micron, micron_to_pix))
            # secax_y11.set_ylabel('y (µm)')

            plt.show()

        return obj.astype(np.complex64)

    def get_probe(self):
        """
        Generate the probe based on the specified type
        """
        if self.probe is not None:
            return self.probe
        else:
            n_of_pixels = self.n_detector_pixels  # Number of pixels in the probe
            z1 = self.source_distance  # Distance from source to object
            w0 = self.probe_gaussian_waist
            z0 = np.pi * w0**2 / self.wavelength  # Rayleigh range

            if z1 < 2 * z0:
                raise Warning("The source distance is within 2x the Rayleigh range. Fresnel Scaling theorem is not valid. Consider increasing the waist size or source distance.")

            if self.probe_type == 'gaussian':
                probe = gaussian_beam(n_of_pixels, self.obj_pixel, z1, w0, self.wavelength)
            elif self.probe_type == 'flat':
                probe = np.ones((n_of_pixels, n_of_pixels), dtype=np.complex64)
            elif self.probe_type == 'disk':
                Y, X = np.ogrid[:n_of_pixels, :n_of_pixels]
                center = (n_of_pixels - 1) / 2
                radius = self.probe_diameter / (2 * self.obj_pixel)
                mask = (X - center)**2 + (Y - center)**2 <= radius**2
                probe = np.zeros((n_of_pixels, n_of_pixels), dtype=np.complex64)
                probe[mask] = 1.0

            probe = probe / np.max(np.abs(probe))  # Normalize the probe amplitude to 1

        if self.probe_desired_counts_per_measurement is not None:
            print(f"Desired counts per measurement: {self.probe_desired_counts_per_measurement:.3e}")

            total_counts_probe_intensity = np.sum(np.abs(probe)**2)
            print(f"Total counts in probe intensity: {total_counts_probe_intensity:.3e}")

            correction_factor = self.probe_desired_counts_per_measurement / total_counts_probe_intensity / probe.size # factor to scale the probe intensity to match the desired counts
            print(f"Correction factor for probe intensity: {correction_factor:.3e}")

            probe = np.sqrt(correction_factor) * probe  # Scale the probe intensity to match desired counts

            checking = np.sum(np.abs(np.fft.fftshift(np.fft.fft2(probe)))**2)  
            print(f"Checking the probe intensity after scaling: {checking:.3e} counts")

        self.probe = probe

        if self.plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax1 = ax[0].imshow(np.abs(self.probe), cmap='viridis')
            ax[0].set_title("Probe Amplitude")
            fig.colorbar(ax1, ax=ax[0])
            ax2 = ax[1].imshow(np.angle(self.probe), cmap='hsv')
            ax[1].set_title("Probe Phase")
            fig.colorbar(ax2, ax=ax[1])
            RGB, sm = get_RGB_probe(self.probe, max_magnitude=np.abs(self.probe).max())
            im = ax[2].imshow(RGB)
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Phase (radians)")
            plt.show()

        return self.probe.astype(np.complex64)

    def get_mask(self):
        if self.mask_type == 'eiger':
            print("Warning: manually setting mask. To be improved.")
            if 1: # get gaps/bad pixels mask
                import xraylab
                eiger_gaps = xraylab.irp.get_eiger_gaps()
                # bad_pixels = tifffile.imread('/home/yrossitonin/data/GINIX/eiger/defects_eiger_run108.tif')
                # dead_pixels_mask = ~create_mask_of_dead_pixels(eiger_gaps, dead_pixels_propability=0.01)
                # mask = np.logical_or(dead_pixels_mask,eiger_gaps)  # combine dead pixels and gaps
                mask = eiger_gaps  # use only eiger gaps
                mask = mask[0:2048, 0:2048]  # crop to match probe shape
            else:
                mask = np.ones(self.probe.shape)

            mask = ~mask

            original_mask = mask.copy()  # keep original mask for later use

            mask = np.roll(mask, shift=(-350,300), axis=(0, 1))  # roll mask to center it

            mask = mask[512:-512, 512:-512]  # crop mask to match probe shape

        else:
            mask = np.ones(self.probe.shape, dtype=bool)
            
        if self.plot:
            fig, ax = plt.subplots(1,2,figsize=(15, 15))
            ax[0].imshow(mask, cmap='gray')
            ax[1].imshow(mask*np.abs(self.probe), cmap='viridis')
            ax[0].set_title("Mask")
            ax[1].set_title("Masked Probe Amplitude")
            
        return mask

    def generate_scan_grid(self, scan_type='raster', **kwargs):
        """
        Generates scan grid points according to the specified scan type.
        
        Parameters:
        scan_type : str
            Type of scan grid to generate. Allowed values are:
                - 'raster': A regular grid.
                - 'raster_noise': A regular grid with added Gaussian noise.
                - 'standard_spiral': An Archimedean spiral starting at (0,0).
                - 'fermat_spiral': A Fermat spiral (using the golden angle) starting at (0,0).
        kwargs: keyword arguments controlling grid parameters.
        
        For 'raster' and 'raster_noise', valid kwargs include:
            num_x (int): Number of points along x (default: 20)
            num_y (int): Number of points along y (default: 20)
            x_min (float): Minimum x (default: -1.0)
            x_max (float): Maximum x (default: 1.0)
            y_min (float): Minimum y (default: -1.0)
            y_max (float): Maximum y (default: 1.0)
            sigma (float): Standard deviation for noise (only for 'raster_noise', default: 0.05)
            
        For 'standard_spiral', valid kwargs include:
            N (int): Total number of points (default: 400)
            theta_max (float): Maximum theta in radians (default: 4*np.pi) 
            max_radius (float): Maximum radius at theta_max (default: 1.0)
            
        For 'fermat_spiral', valid kwargs include:
            N (int): Total number of points (default: 400)
            c (float): Scaling constant (default: 0.05)
            golden_angle (float): Golden angle in radians (default: np.deg2rad(137.508))
        
        Returns:
        points : numpy array of shape (N_points, 2) containing (x, y) coordinates.
        """
        if scan_type == 'raster':
            num_x = kwargs.get('num_x', 20)
            num_y = kwargs.get('num_y', 20)
            x_min = kwargs.get('x_min', -1.0)
            x_max = kwargs.get('x_max', 1.0)
            y_min = kwargs.get('y_min', -1.0)
            y_max = kwargs.get('y_max', 1.0)
            x_raster = np.linspace(x_min, x_max, num_x)
            y_raster = np.linspace(y_min, y_max, num_y)
            X, Y = np.meshgrid(x_raster, y_raster)
            points = np.column_stack((X.ravel(), Y.ravel()))
            return points

        elif scan_type == 'raster_noise':
            # Generate regular grid and then add noise.
            points = self.generate_scan_grid('raster', **kwargs)
            sigma = kwargs.get('sigma', 0.05)
            noise = np.random.normal(loc=0, scale=sigma, size=points.shape)
            return points + noise

        elif scan_type == 'fermat_spiral':
            # Fermat spiral using the golden angle.
            N = kwargs.get('N', 400)
            c = kwargs.get('c', 0.05)
            golden_angle = kwargs.get('golden_angle', np.deg2rad(137.508))
            n = np.arange(N)
            r = c * np.sqrt(n)
            theta = n * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return np.column_stack((x, y))
        else:
            raise ValueError("scan_type must be one of: 'raster', 'raster_noise', 'standard_spiral', 'fermat_spiral'")
        
    def get_scan_positions(self,xmin, xmax, ymin, ymax, n_points, grid_type, fermat_c=0.565,plot=True):
        """
        Generate scan grids, plot them, and return a selected scan grid.
        
        The Fermat spiral grid is parameterized so that the number of points
        remaining inside the rectangular field of view (FOV) is roughly equal
        to the number of points in a raster scan (n_points^2).
        
        Parameters:
        xmin, xmax, ymin, ymax : float
            Limits of the grid in microns.
        n_points : int
            Number of points along each axis for raster-based grids.
        grid_type : str
            Type of grid to return. Options are:
                - 'macro': A theoretical macro scan grid.
                - 'raster': A regular raster scan grid.
                - 'raster_noise' or 'jitter': A noisy raster scan grid.
                - 'fermat': A Fermat spiral scan grid.
        fermat_c : float, optional
            Parameter for the Fermat spiral grid (default is 0.565).
        
        Returns:
        positions : numpy.ndarray
            Array of 2D positions (in meters) for the selected grid.
        """


        # Generate the standard scan grids.
        raster_points = self.generate_scan_grid('raster', num_x=n_points, num_y=n_points, x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)
        
        raster_noise_points = self.generate_scan_grid('raster_noise', num_x=n_points, num_y=n_points,  x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax,  sigma=0.1)
        
        # Heuristic for the Fermat spiral:
        # For a square FOV centered at 0 with limits ±L, the target number of points is n_points^2.
        # Approximately 2/π of the uniformly-distributed points on a circle (that fully contains the square)
        # will fall within the square. Thus, choose N such that:
        #      (2/π)*N ≈ n_points^2   -->   N ≈ (π/2)*n_points^2.
        n_target = n_points ** 2
        N_spiral = int(np.ceil((np.pi / 2) * n_target))
        
        fermat_spiral = self.generate_scan_grid('fermat_spiral', N=N_spiral, c=fermat_c)
        
        # Remove the Fermat spiral points that fall outside the [xmin, xmax] x [ymin, ymax] region.
        mask_outside = ((fermat_spiral[:, 0] < xmin) | (fermat_spiral[:, 0] > xmax) |
                        (fermat_spiral[:, 1] < ymin) | (fermat_spiral[:, 1] > ymax))
        fermat_spiral = np.delete(fermat_spiral, np.where(mask_outside)[0], axis=0)
        
        if plot: # Plot all three scan grids in one row.
            fig, axs = plt.subplots(1, 3, figsize=(22, 6))
            
            # Regular Raster Scan
            axs[0].plot(raster_points[:, 0], raster_points[:, 1], c='blue', marker='o', markersize=5, linestyle='-')
            axs[0].scatter(raster_points[0, 0], raster_points[0, 1], c='green', marker='x', s=100, label='Start')
            axs[0].scatter(raster_points[-1, 0], raster_points[-1, 1], c='red', marker='x', s=100, label='End')
            axs[0].set_title("Regular Raster Scan")
            axs[0].set_xlabel("x (µm)")
            axs[0].set_ylabel("y (µm)")
            axs[0].axis("equal")
            axs[0].grid(True)
            axs[0].legend()
            
            # Raster Scan with Gaussian Noise
            axs[1].plot(raster_noise_points[:, 0], raster_noise_points[:, 1], c='red', marker='o', markersize=5, linestyle='-')
            axs[1].scatter(raster_noise_points[0, 0], raster_noise_points[0, 1], c='green', marker='x', s=100, label='Start')
            axs[1].scatter(raster_noise_points[-1, 0], raster_noise_points[-1, 1], c='red', marker='x', s=100, label='End')
            axs[1].set_title("Raster with Gaussian Noise")
            axs[1].set_xlabel("x (µm)")
            axs[1].set_ylabel("y (µm)")
            axs[1].axis("equal")
            axs[1].grid(True)
            axs[1].legend()
            
            # Fermat Spiral Scan
            axs[2].plot(fermat_spiral[:, 0], fermat_spiral[:, 1], c='green', marker='o', markersize=5, linestyle='-')
            axs[2].scatter(fermat_spiral[0, 0], fermat_spiral[0, 1], c='green', marker='x', s=100, label='Start')
            axs[2].scatter(fermat_spiral[-1, 0], fermat_spiral[-1, 1], c='red', marker='x', s=100, label='End')
            axs[2].set_title("Fermat Spiral Scan")
            axs[2].set_xlabel("x (µm)")
            axs[2].set_ylabel("y (µm)")
            axs[2].axis("equal")
            axs[2].grid(True)
            axs[2].legend()
            
            plt.tight_layout()
            plt.show()
        
            print(f"Raster grid shape: {raster_points.shape}")
            print(f"Raster grid with noise shape: {raster_noise_points.shape}")
            print(f"Fermat spiral grid shape (after cropping): {fermat_spiral.shape}")

        # Select grid type and convert units (from microns to meters).
        if grid_type == 'raster':
            positions = raster_points * 1e-6  # convert µm to meters.
        elif grid_type in ('raster_noise', 'jitter'):
            positions = raster_noise_points * 1e-6  # convert µm to meters.
        elif grid_type == 'fermat':
            positions = fermat_spiral * 1e-6  # convert µm to meters.
        else:
            raise ValueError("grid_type must be one of 'macro', 'raster', 'raster_noise' (or 'jitter'), or 'fermat'")
        
        # Optionally, roll the positions along axis=1 (swapping x and y)
        positions = np.roll(positions, shift=1, axis=1)

        if self.z_values.size == 1:  # If z_values has a single item
            print(f"Adding z={self.z_values[0]*1e3:.2f} mm to positions.")
            z_column = np.full((positions.shape[0], 1), self.z_values[0])
            positions = np.hstack((z_column, positions))
        else:
            new_positions = []
            for z in self.z_values[0:-1]:  # For each z value, create a new set of positions
                print(f"Adding z={z*1e3:.2f} mm to positions.")
                z_column = np.full((positions.shape[0], 1), z)
                three_columns_array = np.hstack((z_column, positions))
                print(three_columns_array.shape)
                new_positions.append(three_columns_array)  # Combine z with Y, X

            positions = np.vstack(new_positions)  # Stack all Z, Y, X positions together
        
        print(f"Selected grid type: {grid_type}. Shape of positions (Z,Y,X): {positions.shape}")

        self.step_size = positions[1,0] - positions[0,0]  # Assuming uniform step size in x

        positions_pxls, positions_pxls_rounded = convert_probe_positions_to_pixel_units(positions, self.obj_pixel)

        if 1: # USE ROUNDED POSITIONS TO GENERATE DATA! i.e. no postion errors will be present
            # When converting from meters to pixels, rounding errors occur. Do this step avoid this rounding of errors to be present in position values in meters
            positions = positions_pxls_rounded*self.obj_pixel

        if 0: # Create another array of positions with errors    
            positions_with_random_error = positions + np.random.uniform(-self.obj_pixel*0.7, self.obj_pixel*0.7, size=positions.shape)
            error = np.sqrt((positions_pxls[:,0]-positions_pxls_rounded[:,0])**2 + (positions_pxls[:,1]-positions_pxls_rounded[:,1])**2)

        if self.plot:
            fig, ax = plt.subplots(1,3, figsize=(15,5))
            ax[0].plot(positions[:,-2]*1e6, positions[:,-1]*1e6, 'o')
            ax[0].set_title('Positions')
            ax[0].set_aspect('equal')
            ax[0].set_xlabel('x [um]')
            ax[0].set_ylabel('y [um]')
            ax[1].plot(positions_pxls[:,-2], positions_pxls[:,-1], 'o')
            ax[1].set_title('Positions in pixels')
            ax[1].set_aspect('equal')
            ax[1].set_xlabel('x [px]')
            ax[1].set_ylabel('y [px]')
            if 1:
                ax[2].plot(positions_pxls[:,-2], positions_pxls[:,-1], 'o',alpha=0.5)
                for (y_orig, x_orig), (y_round, x_round) in zip(positions_pxls[:,1:3], positions_pxls_rounded[:,1:3]):
                    # Note: x is taken from column 1 and y from column 0.
                    ax[2].plot([x_orig, x_round], [y_orig, y_round], 'k-',alpha=0.5)  # 'k-' is a black solid line

            ax[2].plot(positions_pxls_rounded[:,-2], positions_pxls_rounded[:,-1], 'ro',label='Rounded')
            ax[2].set_title('Rounded positions in pixels')
            ax[2].set_aspect('equal')
            ax[2].set_xlabel('x [px]')
            ax[2].set_ylabel('y [px]')
            ax[2].legend(loc='best')
            fig.tight_layout()

        self.positions_pxls = positions_pxls
        self.positions_pxls_rounded = positions_pxls_rounded

        return self.positions_pxls_rounded

    def get_diffraction_data(self):

        if 1: # VIGNETTING:
            self.probe, _ = apply_vignette(self.probe, method='tukey', sigma=3)

            if self.plot:# Plot
                fig, ax = plt.subplots(1, 4, figsize=(20, 4))
                ax0 = ax[0].imshow(np.abs(self.probe),norm=LogNorm())
                ax[0].set_title("Log Probe Amplitude")
                ax1 = ax[1].imshow(np.abs(self.probe), cmap='viridis')
                ax[1].set_title("Probe Amplitude")
                fig.colorbar(ax1, ax=ax[1])
                ax2 = ax[2].imshow(np.angle(self.probe), cmap='hsv')
                ax[2].set_title("Probe Phase")
                fig.colorbar(ax2,ax=ax[2])
                RGB, sm = get_RGB_probe(self.probe, max_magnitude=np.abs(self.probe).max())
                im = ax[3].imshow(RGB)
                divider = make_axes_locatable(ax[3])
                cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust size and pad as needed.
                cbar = fig.colorbar(sm, cax=cax)
                cbar.set_label("Phase (radians)")
                plt.show()

        size_y,size_x = self.probe.shape
        self.ptychogram = np.empty((self.positions.shape[0],size_y, size_x),dtype=np.float32)

        if self.experiment_type == 'PNF' or self.experiment_type == 'PFF' or self.experiment_type == 'CFF':

            # if self.experiment_type == 'PFF':
            #     negative_RoC = phdtoolbox.spherical_RoC(self.obj_pixel, fov, z1, w0, self.wavelength) 
            #     probe = probe*negative_RoC # remove spherical curvature to simulate parallel beam geometry

            for i, (posy, posx) in enumerate(self.positions[:,1:3]):
                if (i+1) % 10 == 0: 
                    print(f"Creating diffraction pattern #{i+1}/{self.positions.shape[0]}", end='\r')
                
                posy, posx = int(posy), int(posx) # Round positions to nearest integer pixel

                obj_roi = self.obj[posy:posy+size_y,posx:posx+size_x]

                wavefront = self.probe*obj_roi

                if 1: #VIGNETTING:
                    wavefront, _ = apply_vignette(wavefront, method='tukey', alpha=0.1)

                self.ptychogram[i] = np.abs(self.propagator(wavefront))**2

                if self.mask is not None: # MASK PIXELS AND ADD NOISE
                    self.ptychogram[i] = self.mask*self.ptychogram[i]

                if self.data_noise is not None:
                    self.ptychogram[i] = np.random.poisson(self.ptychogram[i])

            if self.plot:
                plt.figure(figsize=(6,6))
                plt.imshow(self.ptychogram[self.ptychogram.shape[0]//2], norm=LogNorm())
                plt.title("Simulated diffraction pattern (log scale)")
                plt.colorbar()
                plt.show()

        elif self.experiment_type == 'CMD':

            for i, (zj,posy, posx) in enumerate(self.positions):
                if (i+1) % 10 == 0: 
                    print(f"Creating diffraction pattern #{i+1}/{self.positions.shape[0]}", end='\r')
                
                posy, posx = int(posy), int(posx) # Round positions to nearest integer pixel

                obj_roi = self.obj[posy:posy+size_y,posx:posx+size_x]

                wavefront = self.probe*obj_roi

                if 1: #VIGNETTING:
                    wavefront, _ = apply_vignette(wavefront, method='tukey', alpha=0.1)

                wavefront = propagate_multiplane(obj_roi,self.probe, self.obj_pixel,self.z_values[0],zj,self.z_values[-1],self.wavelength)
                self.ptychogram[i] = np.abs(wavefront)**2

                if self.mask is not None: # MASK PIXELS AND ADD NOISE
                    self.ptychogram[i] = self.mask*self.ptychogram[i]

                if self.data_noise is not None:
                    self.ptychogram[i] = np.random.poisson(self.ptychogram[i])

            if self.plot:
                plt.figure(figsize=(6,6))
                plt.imshow(self.ptychogram[self.ptychogram.shape[0]//2], norm=LogNorm())
                plt.title("Simulated diffraction pattern (log scale)")
                plt.colorbar()
                plt.show()


        if self.data_upsampling is not None:
            if self.data_upsampling > 1: # UPSAMPLE DIFFRACTION PATTERN
                self.ptychogram = upsample_complex_image(self.ptychogram,self.data_upsampling)

        if 0: # Remove scan points if wanted
            indices_to_keep = np.random.choice(positions.shape[0], 200, replace=False)
            indices_to_keep = np.sort(indices_to_keep)
            positions = positions[indices_to_keep]
            positions_pxls = positions_pxls[indices_to_keep]
            positions_pxls_rounded = positions_pxls_rounded[indices_to_keep]
            positions_with_random_error = positions_with_random_error[indices_to_keep]
            self.ptychogram = self.ptychogram[indices_to_keep]

        if self.output_filepath is not None: # SAVE TO HDF5

            parent_dir = os.path.dirname(self.output_filepath)
            os.makedirs(parent_dir, exist_ok=True)


            print(self.ptychogram.dtype, self.ptychogram.shape)
            with h5py.File(self.output_filepath,'w') as f:
                f.create_dataset('metadata/wavelength',data=self.wavelength)
                f.create_dataset('metadata/energy',data=self.energy)
                f.create_dataset('metadata/detector_pixel',data=self.detector_pixel_size)
                # f.create_dataset('metadata/z1',data=self.z1)
                # f.create_dataset('metadata/z2',data=self.z2)
                # f.create_dataset('metadata/delta',data=self.delta)
                # f.create_dataset('metadata/beta',data=self.beta)
                f.create_dataset('model_obj',data=self.obj)   
                f.create_dataset('model_probe',data=self.probe)
                f.create_dataset('positions',data=self.positions)
                f.create_dataset('positions_x',data=self.positions[:,1])
                f.create_dataset('positions_y',data=self.positions[:,0])
                f.create_dataset('mask',data=self.mask)
                f.create_dataset('ptychogram',data=self.ptychogram)

            print(f"Simulated dataset saved to {self.output_filepath}")











