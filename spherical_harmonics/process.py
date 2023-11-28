import pyshtools as pysh
import numpy as np
import scipy
from validation import assert_shape

def S20RTS_to_pyshtools(clm: pysh.SHCoeffs, sh_coeffs: np.ndarray) -> pysh.SHCoeffs:
    """
    Takes in a 1D NumPy array of S20RTS spherical harmonic coefficients and returns a 3D NumPy array of pyshtools spherical harmonic coefficents.

    Parameters
    ----------
    clm : pysh.SHCoeffs(cphase=1, normalization='ortho')
        Non-initialised spherical harmonic coefficient class.
    sh_coeffs : np.ndarray(dtype=float, ndim=1)
        S20RTS spherical harmonic coefficients.

    Returns
    -------
    clm : pysh.SHCoeffs(cphase=1, normalization='ortho')
        Initialised spherical harmonic coefficient class.
    """

    '''
    For positive orders, the coefficients are stored in the following order:

    c_{0, 0}           0,           0,       ...,       0
    c_{1, 0}        c{1, 1},        0,       ...,       0
    c_{2, 0}        c{2, 1},     c{2, 2},    ...,       0
    c_{3, 0}        c{3, 1},     c{3, 2},    ...,       0
        .              .            .         .        .        
        .              .            .         .        .         
        .              .            .         .        .         
    c_{l_max, 0}, c{l_max, 1}, c{l_max, 2}, ..., c{l_max, l_max}


    For negative orders, the coefficients are stored in the following order:

    0       0,            0,        ...,          0
    0   c_{1, -1},        0,        ...,          0
    0   c_{2, -1},    c_{2, -2},    ...,          0
    0   c_{3, -1},    c_{3, -2},    ...,          0
    .      .              .           .           .        
    .      .              .           .           .         
    .      .              .           .           .         
    0 c_{l_max, -1}, c_{l_max, -2}, ..., c_{l_max, -l_max}

    '''

    assert_shape(sh_coeffs, ((clm.lmax + 1) ** 2,))
    # Positive orders
    for l in range(clm.lmax + 1):
        index = l * (l + 1)
        for order in range(l + 1):
            clm.coeffs[0, l, order] = sh_coeffs[index + order]

    # Negative orders
    for l in range(1, clm.lmax + 1):
        index = l * (l + 1)
        for order in range(1, l + 1):
            clm.coeffs[1, l, order] = sh_coeffs[index - order]

    return clm

def shear_wave_to_density(shear_wave_coeffs):
    """
    This function takes in shear wave velocity anomaly coefficients and returns density anomaly
    coefficients.

    Parameters
    ----------
    shear_wave_coeffs : np.ndarray(dtype=float, ndim=2)
        Array of shear wave velocity anomaly coefficients.

    Returns
    -------
    density_coeffs : np.ndarray(dtype=float, ndim=2)
        Array of density anomaly coefficients using a scaling factor.
    """
    density_coeffs = shear_wave_coeffs * 0.1
    return density_coeffs

def calculate_observable(observable: str, l: int, l_max:int, density_anomaly_sh_lm: np.ndarray,
                         radius_arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    This function calculates the observable for a given degree.

    The set of observables is:
        {
            'surftopo': Dynamic Topography,
            'geoid': Geoid,
            'gravity': Gravity Anomaly
            'cmbtopo': CMB Topography
        }

    Parameters
    ----------
    observable : str
        The observable to calculate.
    l : int
        The degree to calculate the observable at.
    density_anomaly_sh_lm : np.ndarray(dtype=float, ndim=4)
        Array of density anomaly spherical harmonic coefficients.
    radius_arr : np.ndarray(dtype=float, ndim=1)
        Array of radii values.
    kernel : np.ndarray(dtype=float, ndim=1)
        Array of kernel values.

    Returns
    -------
    integral : np.ndarray(dtype=float, ndim=1)
        Array of integrated density anomalies x kernel.
    """
    assert_shape(density_anomaly_sh_lm, (len(radius_arr), 2, l_max+1, l_max+1))
    density_sh_l = density_anomaly_sh_lm[:, :, l, :]
    # integrand is (radial_points x (2*l_max + 1))
    integrand = np.concatenate(
                        (density_sh_l[:, 1, :] * kernel.reshape(-1, 1),
                               density_sh_l[:, 0, :] * kernel.reshape(-1, 1)),
                               axis=1
                               )

    # integrate along the radius for each order
    integral = scipy.integrate.simpson(integrand, radius_arr, axis=0)
    assert_shape(integral, (2 * (l_max + 1),))
    return integral
