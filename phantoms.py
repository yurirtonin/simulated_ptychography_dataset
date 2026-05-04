import numpy as np
import matplotlib.pyplot as plt

import xraydb
from hotopy.datasets import dicty, dicty_multi, world, beads, radiodurans,macrophage, world_holograms, spider

def get_delta_beta_xraydb(material,energy):

    if material == 'Ni': # nickel
        density = 8.908 # g/cm^3
    elif material == 'Si': # silicon
        density = 2.329002
    elif material == 'W': # tungsten
        density = 19.25
    elif material == 'Au': # gold
        density = 19.3
    elif material == 'Ta': # tantalum
        density = 16.4
    elif material == 'H50C40N9O10S1':
        density = 1.35 # protein of empirical formula H50C30N9O10S1 and density 1.35 g/cm3. See Howells An assessment of the resolution limitation due to radiation-damage in X-ray diffraction microscopy
        
    delta, beta, attenuation_length = xraydb.xray_delta_beta(material,density,energy)
    return delta, beta 

def get_betadelta_plot(element='H50C40N9O10S1', energy_keV=8.0, max_energy=20e3, logscale=True,plot=True):
    """
    Plot the refractive index (delta and beta) and their ratio as a function of energy.
    """
    energy_range = np.linspace(30, max_energy, 10000)  # Energy range from 30 eV to max_energy eV
    delta_values, beta_values = get_delta_beta_xraydb(element, energy_range) 

    # Calculate intercepts at energy_keV
    idx = np.argmin(np.abs(energy_range - energy_keV*1e3)) # Find index where energy_range - max_energy is closest to 0
    beta_at_e = float(beta_values[idx])
    delta_at_e = float(delta_values[idx])
    ratio_at_e = float(beta_at_e / delta_at_e)

    if plot:
        # Plot
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
        ax1.plot(energy_range*1e-3, beta_values, label=r'$\beta$')
        ax1.plot(energy_range*1e-3, delta_values, label=r'$\delta$')
        ax1.axvline(x=energy_keV, color='r', linestyle='--', label='Used energy')
        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('Refractive Index')
        if logscale:
            ax1.set_yscale('log')
            ax1.set_xscale('log')
        ax1.grid()

        # Markers for Beta and Delta at intercept
        ax1.plot(energy_keV, beta_at_e, 'o', color='tab:blue', markersize=5)
        ax1.plot(energy_keV, delta_at_e, 'o', color='tab:orange', markersize=5)

        ax2 = ax1.twinx()
        line2, = ax2.plot(energy_range*1e-3, beta_values/ delta_values, color='g', label=r'$\beta/\delta$ Ratio')
        ax2.set_ylabel(r'$\beta/\delta$ Ratio', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_yscale('log')

        # Marker for Beta/Delta Ratio at intercept
        ax2.plot(energy_keV, ratio_at_e, 'o', color='g', markersize=5)

        # Combine legends from both axes and place outside the plot area
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = [line2], [line2.get_label()]
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

        # Add intercept values to the title
        ax1.set_title(
            fr"Intercepts at {energy_keV:.2f} keV: "
            fr"$\beta$={beta_at_e:.2e}, $\delta$={delta_at_e:.2e}, $\beta/\delta$={ratio_at_e:.2e}"
        )

        plt.tight_layout()
        plt.show()    

    return beta_at_e, delta_at_e, ratio_at_e

class Phantom:
    def __init__(self, source='camera_gravel',magnitude=None, phase=None, shape=None, normalize=True):
        """
        source: 'camera_gravel' or 'dicty' # only used if magnitude and phase are None
        magnitude: numpy array or None, magnitude image to use (if provided)
        phase: numpy array or None, phase image to use (if provided)
        shape: tuple or None, resize output to this shape if not None
        normalize: if True, scale images to [0, 1]
        """

        if magnitude is not None or phase is not None:
            if magnitude is None or phase is None:
                raise ValueError("Both magnitude and phase must be provided if one is given.")
            image1 = magnitude
            image2 = phase
        else:
            if source == 'camera_gravel':
                from skimage import data
                image1 = data.camera()
                image2 = data.gravel()
            elif source == 'dicty':
                image1 = dicty()#[0::2, 0::2]
                image2 = image1.copy()
            else:
                raise ValueError("Unknown source. Use 'camera_gravel' or 'dicty'.")

        print(f"Original image shape: {image1.shape} and {image2.shape}")
        if shape is not None:
            from skimage.transform import resize
            image1 = resize(image1, shape, preserve_range=True)
            image2 = resize(image2, shape, preserve_range=True)
            print(f"Resized image shape: {image1.shape} and {image2.shape}")

        if normalize:
            image1 = image1 / np.max(image1)
            image2 = image2 / np.max(image2)

        self.magnitude = image1
        self.phase = image2
        self.complex_image = image1 * np.exp(1j * image2)

    def get_magnitude(self):
        return self.magnitude

    def get_phase(self):
        return self.phase

    def get_complex(self):
        return self.complex_image
    
    def get_complex_with_betadelta(self, element='H50C40N9O10S1', energy_keV=None,wavelength=None,membrane_value=0.653, phantom_thickness=10e-6,max_energy=20e3,logscale=True,plot=True):

        speed_of_light = 299792458        # Speed of Light [m/s]
        planck         = 4.135667662E-18  # Plank constant [keV*s]
        if energy_keV is None:
            energy_keV = planck * speed_of_light / wavelength
        if wavelength is None:
            wavelength = planck * speed_of_light / energy_keV

        k_wavector = 2*np.pi/wavelength
        beta_at_e, delta_at_e, ratio_at_e = get_betadelta_plot(element, energy_keV, max_energy, logscale,plot=plot)

        # print(f"Wavelength = {wavelength:.3e} m")
        # print(rf"Energy = {energy_keV*1e3:.0f} eV, delta = {delta_at_e:.3e}, beta = {beta_at_e:.3e}")

        normalized_image = self.magnitude
        image = normalized_image/membrane_value # Normalizing the phantom so that membrane pixel value is 1
        complex_phantom = (-delta_at_e*image + 1j*beta_at_e*image)
        complex_projection = np.exp(1j*k_wavector*phantom_thickness*complex_phantom) # transmission function
        return complex_projection, beta_at_e,delta_at_e,ratio_at_e