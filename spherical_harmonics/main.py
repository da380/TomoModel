import numpy as np
import pyshtools as pysh
import util
import matplotlib.pyplot as plt
import pygmt

# Set the maximum spherical harmonic degree.
l_max = 20

# Read in spherical harmonic coefficients
file_path = '../out/output.txt'
v_s_sh_coeffs, radius, ref_density, ref_shear_velocity = util.preprocess(file_path)

# Interpolate the radius and kernel to fit S20RTS format
# Create a sample 20-element array

# Convert shear wave velocity coefficients to density coefficients.
density_sh_coeffs = util.shear_wave_to_density(v_s_sh_coeffs)

# Initialise a pyshtools SHCoeffs class for each radius.
density_coeffs_clm = np.array([pysh.SHCoeffs.from_zeros(lmax=l_max, kind='real',
                                                        normalization='ortho') for i, v in
                          enumerate(radius)])

# Use the Condon-Shortley phase convention.
for i, clm in enumerate(density_coeffs_clm):
    clm.cphase = -1
    clm = util.S20RTS_to_pyshtools(sh_coeffs=density_sh_coeffs[i, :], clm=clm)


density_coeffs_array = np.array([clm.coeffs for clm in density_coeffs_clm])


# testing the integrand function
test_coeffs = util.generate_test_coeffs(len(radius), l_max=l_max)
test_kernel = np.ones(len(radius))

kernel_root = '../data/kernels/'
visc_name = 'const_visc/'
kernel_type = 'surftopo'

clm_DT_array = np.zeros((2, l_max+1, l_max+1))

for degree in range(1, l_max+1):
    kernel_path = kernel_root + visc_name + kernel_type + util.leading_zeros(degree)

    radius_kernel, kernel = util.parse_file(kernel_path)
    radius_kernel *= 1.e3

    # Interpolate
    kernel = np.interp(radius, radius_kernel, kernel)

    integral = util.integrate(l=degree, density_sh_lm=density_coeffs_array, kernel=kernel,
                          radius_arr=radius)

    m_negative = integral[:l_max+1]
    m_positive = integral[l_max+1:]

    # Positive
    clm_DT_array[0][degree] = m_negative
    # Negative
    clm_DT_array[1][degree] = m_positive

clm_DT = pysh.SHCoeffs.from_array(clm_DT_array, normalization='ortho')
clm_DT.cphase = 1
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Plot the data
grid_data = clm_DT.expand().data
im = ax.imshow(grid_data, cmap='viridis', extent=[-180, 180, -90, 90])

ax.set_ylabel(r'Latitude ($^\circ$)')
ax.set_xlabel(r'Longitude ($^\circ$)')

# Add colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Dynamic Topography (m)')
plt.show()

fig, ax = clm_DT.plot_spectrum()
plt.show()
