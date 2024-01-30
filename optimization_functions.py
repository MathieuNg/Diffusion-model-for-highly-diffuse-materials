# Copyright (C) 2024 - Mathieu Nguyen

# This program is free software: you can redistribute it and/or modify 
# it under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or 
# (at your option) any later version. 

# This program is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU General Public License for more details. 

# You should have received a copy of the GNU General Public License 
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from scipy.optimize import curve_fit

def get_normalized_radius(r, data):
    """
    Computes the normalization of reflectance data profiles and returns also
    the distance for which the maximum is located.

    Inputs:
    r (array-like): vector of radial distance given in millimeters (from 0 to 20 mm).
    data (array-like): vector of the reflectance profile for a specific colour
    channel and record.

    Outputs:
    r0 (float): value of the radial distance from entry point in the
    material for which the reflectance signal is at maximum.
    data (array-like): normalized reflectance profile.
    """
    data = data / np.max(data)

    cond = np.where(data == 1)[0]
    r0 = r[cond[0]]

    return r0, data

def fit_function_E(A, r, data, transport):
    """
    Nonlinear least-square fitting of the diffusion model by only having the
    efficient coefficient as a variable to estimate.

    Inputs:
    A (float): diffuse Fresnel parameter for internal reflections in the
    material.
    r (array-like): vector of radial distance given in millimeters (from 0 to 20 mm).
    data (array-like): vector of the reflectance profile for a specific colour
    channel and record.
    transport (float): transport coefficient in mm-1 given as a starting point.

    Output:
    efficient (float): estimate of the efficient coefficient in mm-1.
    """
    # Normalization of data
    r0, data = get_normalized_radius(r, data)

    # Fitting the model
    def model_fit_function(r, efficient):
        return model_function_E(efficient, transport, A, r) / model_function_E(efficient, transport, A, r0)

    p0 = [3]  # Starting point
    bounds = (0, 10)  # Bounds for efficient coefficient
    efficient, _ = curve_fit(model_fit_function, r, data, bounds=bounds, p0=p0, maxfev=1000)

    return efficient[0]

def fit_function_T(A, r, data, efficient):
    """
    Nonlinear least-square fitting of the diffusion model by only having the
    transport coefficient as a variable to estimate.

    Inputs:
    A (float): diffuse Fresnel parameter for internal reflections in the
    material.
    r (array-like): vector of radial distance given in millimeters (from 0 to 20 mm).
    data (array-like): vector of the reflectance profile for a specific colour
    channel and record.
    efficient (float): efficient coefficient in mm-1 given as a starting point.

    Output:
    transport (float): estimate of the transport coefficient in mm-1.
    """
    # Normalization of data
    r0, data = get_normalized_radius(r, data)

    # Fitting the model
    def model_fit_function(r, transport):
        return model_function_T(transport, efficient, A, r) / model_function_T(transport, efficient, A, r0)

    p0 = [3]  # Starting point
    bounds = (0, 10)  # Bounds for transport coefficient
    transport, _ = curve_fit(model_fit_function, r, data, bounds=bounds, p0=p0, maxfev=1000)

    return transport[0]

def model_function_E(sigEff, sigT, A, r):
    """
    Returns the diffusion model following the work of Farrell et al., 
    “A diffusion theory model of spatially resolved, steady-state diffuse
    reflectance for the noninvasive determination of tissue optical 
    properties in vivo,” Med. Phys. 19, 879–888 (1992).
    This equation is parameterized with the transport (sigT) and efficient
    (sigEff) coefficients to be adapted to our inversion method.
    Use this function when sigEff is the parameter to estimate while sigEff is
    known.

    Inputs:
    sigEff (float): efficient coefficient in mm-1 unknown and to be estimated.
    sigT (float): transport coefficient in mm-1 with value given.
    A (float): diffuse Fresnel parameter for internal reflections in the
    material.
    r (array-like): vector of radial distance given in millimeters (from 0 to 20
    mm).

    Output:
    f (array-like): diffusion model to use for fitting.
    """
    r1 = np.sqrt((1./sigT)**2 + r**2)
    r2 = np.sqrt(((1./sigT)*(1+4*A/3))**2 + r**2)

    f = ((sigEff + 1./r1)*np.exp(-sigEff*r1)/r1**2 +
         (1 + 4*A/3)*(sigEff + 1./r2)*np.exp(-sigEff*r2)/r2**2)

    return f

def model_function_T(sigT, sigEff, A, r):
    """
    Returns the diffusion model following the work of Farrell et al., 
    “A diffusion theory model of spatially resolved, steady-state diffuse
    reflectance for the noninvasive determination of tissue optical 
    properties in vivo,” Med. Phys. 19, 879–888 (1992).
    This equation is parameterized with the transport (sigT) and efficient
    (sigEff) coefficients to be adapted to our inversion method.
    Use this function when sigT is the parameter to estimate while sigEff is
    known.

    Inputs:
    sigT (float): transport coefficient in mm-1 unknown and to be estimated.
    sigEff (float): efficient coefficient in mm-1 with value given.
    A (float): diffuse Fresnel parameter for internal reflections in the
    material.
    r (array-like): vector of radial distance given in millimeters (from 0 to 20
    mm).

    Output:
    f (array-like): diffusion model to use for fitting.
    """
    r1 = np.sqrt((1./sigT)**2 + r**2)
    r2 = np.sqrt(((1./sigT)*(1+4*A/3))**2 + r**2)

    f = ((sigEff + 1./r1)*np.exp(-sigEff*r1)/r1**2 +
         (1 + 4*A/3)*(sigEff + 1./r2)*np.exp(-sigEff*r2)/r2**2)

    return f

