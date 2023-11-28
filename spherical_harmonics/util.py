import numpy as np
from  scipy.integrate import simpson
import os

def leading_zeros(degree):
    return str(degree).zfill(3)


def parse_file(path, number_of_columns=2):
    with open(path, "r") as file:
        # Initialize empty lists to store the data
        radial_data = []
        kernel_data = []

        # Read and process each line
        for line in file:
            # Split the line into columns using space as the delimiter
            columns = line.strip().split()

            # Check if there are two columns
            if len(columns) == 2:
                try:
                    # Convert the values to floats and append to respective lists
                    radial_data.append(float(columns[0]))
                    kernel_data.append(float(columns[1]))
                except ValueError:
                    print("Error: Invalid data format on line:", line)
            else:
                print("Error: Invalid data format on line:", line)
    # convert to 2 dimensional numpy array where rows are kernel data and columns are radial data
    return np.array(radial_data), np.flip(np.array(kernel_data))


def preprocess(file_path):
    """
    Preprocesses the S20RTS spherical harmonic coefficients given a file path.

    :param str file_path: The path to the S20RTS spherical harmonic coefficients.
    :return np.ndarray sh_coeffs: 2D np array of S20RTS spherical harmonic coefficients.
    :return np.ndarray radius: 1D np array of radii at which the spherical harmonic coefficients are evaluated.
    :return np.ndarray ref_density: 2D np array of reference density values at which the spherical harmonic coefficients are evaluated.
    :return np.ndarray ref_shear_velocity: 2D np array of reference shear wave velocity values at which the spherical harmonic coefficients are evaluated.

    """

    # Load in the S20RTS spherical harmonic coefficients.
    S20RTS_data = np.loadtxt(file_path)

    radius = S20RTS_data[:, 0]
    ref_density = S20RTS_data[:, 1]
    ref_shear_velocity = S20RTS_data[:, 2]
    sh_coeffs = S20RTS_data[:, 3:]

    size = sh_coeffs.shape[1]

    # First half of the sh_coeffs array is the real part, second half is the imaginary part.
    real_sh_coeffs = sh_coeffs[:, 0: int(size / 2)]
    imag_sh_coeffs = sh_coeffs[:, int(size / 2):]

    # Convert into complex numbers.
    sh_coeffs = real_sh_coeffs + 1j * imag_sh_coeffs

    return sh_coeffs, radius, ref_density, ref_shear_velocity


def S20RTS_to_pyshtools(clm, sh_coeffs):
    """
    Takes in a 1D NumPy array of S20RTS spherical harmonic coefficients and returns a 3D NumPy array of pyshtools spherical harmonic coefficents.


    :param pysh.SHCoeffs clm: Non-initialised spherical harmonic coefficient class.
    :param np.ndarray sh_coeffs: S20RTS spherical harmonic coefficients.
    :return pysh.SHCoeffs clm: Initialised spherical harmonic coefficient class.

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


def integrate(l, density_sh_lm, radius_arr, kernel):
    """

    :param int l: Degree to evaluate integral at.
    :param np.ndarray density_sh_lm: 4D array of spherical harmonic coefficients evaluated at
    a degree. (radial points, +-order, l_max+1)
    :param np.ndarray radius_arr: 1D array of radius values.
    :param np.ndarray kernel: 1D array of kernel as function of depth.

    :returns np.ndarray integral: 3D array of integrated density anomalies x kernel. (radial
    points, +-order, l_max+1
    """

    density_sh_l = density_sh_lm[:, :, l, :]
    # integrand is (radial_points x (2*l_max + 1))
    integrand = np.concatenate(
                        (density_sh_l[:, 1, :] * kernel.reshape(-1, 1),
                               density_sh_l[:, 0, :] * kernel.reshape(-1, 1)),
                               axis=1
                               )

    # integrate along the radius for each order
    integral = simpson(integrand, radius_arr, axis=0)
    return integral

def generate_test_coeffs(num_radial, l_max):
    """
    Returns a 3D np array of test coefficients of the form

    [
    [00 00 00 ... ... ... ... 00]
    [10 11 00 ... ... ... ... 00]
    [20 21 22 ... ... ... ... 00]
    [30 31 32 ... ... ... ... 00]
    ...
    [l0 ... ... ... .. ... ... ll],

    [00 00 00 ... ... ... ... 00],
    [00 -10 -11 ... ... ... ... 00],
    [00 -20 -21 ... ... ... ... 00],
    ...
    [00 -l0 ... ... ... .. ... ... -ll]
    ] x number of radial points.
    """

    # radial points x 2 x l_max+1 x l_max+1
    mat = np.array([[[[0 for i in range(l_max+1)] for j in range(l_max+1)] for k in range(2)]
                     for r in (range(num_radial))])

    # Positive orders.
    for r in range(num_radial):
        for l in range(l_max+1):
            for m in range(l+1):
                mat[r][0][l][m] = int(str(l) + str(m))

    # Negative orders.
    for r in range(num_radial):
        for l in range(l_max+1):
            for m in range(1, l+1):
                mat[r][1][l][m] = -int(str(l) + str(m))


    return mat

def shear_wave_to_density(shear_wave_coeffs):
    """
    Returns the density anomaly coefficients given the shear wave velocity anomaly coefficients.
    :params np.ndarray shear_wave_coeffs: 2D np array of shear wave velocity anomaly coefficients.
    :return np.ndarray density_coeffs: 2D np array of density anomaly coefficients.
    """
    density_coeffs = shear_wave_coeffs * 0.1
    return density_coeffs
