# TomoModel

ReadS20RTS takes in models in the S20RTS/S40RTS format and prints out to an ascii file the spherical harmonic coefficients for the model at a discrete set of radii. 

Inputs: [Name of spherical reference model in deck format] [Name of 3D model file in S20RTS format] [name for the output file]

Within the data directory examples of the two input files can be found (specifically, PREM and S20RTS). Each line of the output file contains the following:

radius (m)  | reference density (kg / m^{3}) | reference shear velocity (m / s) | spherical harmonic coefficients for dlnvs 

For each radius the spherical harmonic coefficients are listed in the following order, denoting the function by f for simplicity:

*
\mathrm{Re} f_{00}
*
