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


import  numpy as np
import scipy
from data_processing_functions import *
from optimization_functions import fit_function_E, fit_function_T

def compute_channel_fitting_init(A, r, sep, data):
    """
    Computes the inversion method for the Part 1 Initialization. The computation is done for a specific channel and record.

    Args:
    - A (float): diffuse Fresnel parameter for internal reflections in the material.
    - r (ndarray): vector of radial distance given in millimeters (from 0 to 20 mm).
    - sep (int): integer to separate the reflectance data.
    - data (ndarray): vector of the reflectance profile for a specific color channel and record.

    Returns:
    - transport (ndarray): transport coefficients estimated on the first part of the reflectance curve.
    - efficient (ndarray): efficient coefficients estimated on the second part of the reflectance curve.
    """
    # Separation of vector for distance and reflectance data
    r1, r2 = get_array_separated(r, sep)
    data1, data2 = get_array_separated(data, sep)

    # Both estimations are assumed to be independent from each other, which
    # explains the use of assigned values to 1 for efficient and transport
    # coefficients.

    # Initial estimation of the transport coefficient assuming a value for
    # efficient coefficient in this part of the curve.
    sigEff_0 = 1
    transport = fit_function_T(A, r1, data1, sigEff_0)

    # Initial estimation of the efficient coefficient assuming a value for
    # transport coefficient in this part of the curve.
    sigT_0 = 1
    efficient = fit_function_E(A, r2, data2, sigT_0)

    return transport, efficient

def compute_channel_fitting_loop(A, r, sep, data, eff0, trp0):
    """
    Computes the inversion method to reach convergence on the final estimates
    of the optical properties. It requires values from the initialization
    part.

    Inputs:
    A (double): diffuse Fresnel parameter for internal reflections in the
    material.
    r (array-like): vector of radial distance given in millimeters (from 0 to 20 mm).
    sep (int): integer to separate the reflectance data.
    data (array-like): vector of the reflectance profile for a specific colour
    channel and record.
    eff0 (double): previous iteration of efficient coefficient in mm-1.
    trp0 (double): previous iteration of transport coefficient in mm-1.

    Outputs:
    transport (double): estimate of transport coefficient in mm-1.
    efficient (double): estimate of efficient coefficient in mm-1.
    """
    # Proceeding to the separation of radial distances and reflectance profiles data.
    r1, r2 = get_array_separated(r, sep)
    data1, data2 = get_array_separated(data, sep)

    # Computing new estimates of optical coefficients with previous iterated values.
    transport = fit_function_T(A, r1, data1, eff0)
    efficient = fit_function_E(A, r2, data2, trp0)

    return transport, efficient

def compute_inversion_method1(dataFilename, textFilename, matFilename, statFilename, numIterations, eta):
    # Reading data from TLS text file that must be exported from the software
    # first. All records for each channel are extracted and stored in the
    # following variables.
    dataR, dataG, dataB = read_data_TLS(dataFilename)
    numRecords = dataR.shape[1]

    # Diffuse Fresnel parameter varying depending on the material and the real
    # part of the refractive index
    A = get_diffuse_fresnel(eta)

    # Modelisation of the NMOS sensor array. Its length is of 20 millimeters
    # and it is needed to compute the diffusion model and obtain estimates of
    # optical coefficients in the proper unit.
    r = np.linspace(0, 20, 512)

    # The following part is the algorithm of the Inversion method. It is
    # divided into two parts. First is the initialisation of the loop process
    # with a first estimation of optical coefficients. Second part is the loop to
    # reach convergence and final values of optical coefficients. The method
    # focuses on estimating efficient (eff) and transport (trp) coefficient,
    # which are then converted to absorption (abs) and reduced scattering (sca)
    # coefficients.

    # Part 1: Initialization
    # Creating initial guesses
    eff0 = np.zeros((numRecords, 3))
    trp0 = np.zeros((numRecords, 3))

    # Loop over all records
    for rec in range(numRecords):
        # RED Channel
        proDataR, proDistanceR = get_preprocessed_data(dataR[:, rec], r)
        sepR = get_threshold_index(proDistanceR, proDataR)

        trp, eff = compute_channel_fitting_init(A, proDistanceR, sepR, proDataR)
        trp0[rec, 0] = trp
        eff0[rec, 0] = eff

        # GREEN Channel
        proDataG, proDistanceG = get_preprocessed_data(dataG[:, rec], r)
        sepG = get_threshold_index(proDistanceG, proDataG)

        trp, eff = compute_channel_fitting_init(A, proDistanceG, sepG, proDataG)
        trp0[rec, 1] = trp
        eff0[rec, 1] = eff

        # BLUE Channel
        proDataB, proDistanceB = get_preprocessed_data(dataB[:, rec], r)
        sepB = get_threshold_index(proDistanceB, proDataB)

        trp, eff = compute_channel_fitting_init(A, proDistanceB, sepB, proDataB)
        trp0[rec, 2] = trp
        eff0[rec, 2] = eff

    # After this loop, the first initial estimates for the optical properties
    # are known and will be used in the second part to reach convergence.
    # Instead of having a random guess for sigT and sigEff, previous iterations
    # would be used to refine the final values.

    # Continue with Part 2...
    # Part 2: Iteration process for convergence
    # Storing data for each iteration
    saveEff = np.zeros((eff0.shape[0], eff0.shape[1], numIterations+1))
    saveTrp = np.zeros((trp0.shape[0], trp0.shape[1], numIterations+1))
    
    # Initialization of saving matrices
    saveEff[:, :, 0] = eff0
    saveTrp[:, :, 0] = trp0
    
    # Iteration loop
    for i in range(1, numIterations+1):
        print(f"Iteration #{i}")
    
        eff = saveEff[:, :, i-1]
        trp = saveTrp[:, :, i-1]
    
        # Initialization
        effInt = np.zeros((numRecords, 3))
        trpInt = np.zeros((numRecords, 3))
    
        for rec in range(numRecords):
            # RED Channel
            proDataR, proDistanceR = get_preprocessed_data(dataR[:, rec], r)
            sepR = get_threshold_index(proDistanceR, proDataR)
    
            EFF = eff[rec, 0]
            TRP = trp[rec, 0]
            transport, efficient = compute_channel_fitting_loop(A, proDistanceR, sepR, proDataR, EFF, TRP)
            effInt[rec, 0] = efficient
            trpInt[rec, 0] = transport
    
            # GREEN Channel
            proDataG, proDistanceG = get_preprocessed_data(dataG[:, rec], r)
            sepG = get_threshold_index(proDistanceG, proDataG)
    
            EFF = eff[rec, 1]
            TRP = trp[rec, 1]
            transport, efficient = compute_channel_fitting_loop(A, proDistanceG, sepG, proDataG, EFF, TRP)
            effInt[rec, 1] = efficient
            trpInt[rec, 1] = transport
    
            # BLUE Channel
            proDataB, proDistanceB = get_preprocessed_data(dataB[:, rec], r)
            sepB = get_threshold_index(proDistanceB, proDataB)
    
            EFF = eff[rec, 2]
            TRP = trp[rec, 2]
            transport, efficient = compute_channel_fitting_loop(A, proDistanceB, sepB, proDataB, EFF, TRP)
            effInt[rec, 2] = efficient
            trpInt[rec, 2] = transport
    
        # Saving results for next iteration
        saveEff[:, :, i] = effInt
        saveTrp[:, :, i] = trpInt
    
    # Converting from (transport, efficient) to (absorption, reduced scattering)
    # by taking the mean of all estimates for transport and efficient coefficients (method 1)
    finalEff = np.mean(saveEff[:, :, -1], axis=0)
    finalTrp = np.mean(saveTrp[:, :, -1], axis=0)
    finalSca, finalAbs = get_optical_properties(finalTrp, finalEff)
    
    # Writing processes in the .txt file and .mat file
    with open(textFilename, 'w') as resultsID:
        resultsID.write(f'Abs (RGB): {finalAbs[0]:.3e} {finalAbs[1]:.3e} {finalAbs[2]:.3e}\n')
        resultsID.write(f'Sca (RGB): {finalSca[0]:.3e} {finalSca[1]:.3e} {finalSca[2]:.3e}\n\n')
    
    # Saving data to .mat file for statistical studies of optical properties
    statSca, statAbs = get_optical_properties(saveTrp[:, :, -1], saveEff[:, :, -1])
    scipy.io.savemat(statFilename, {'statAbs': statAbs, 'statSca': statSca})
    
    # Saving to .mat file
    scipy.io.savemat(matFilename, {'finalAbs': finalAbs, 'finalSca': finalSca})

def compute_inversion_method2(dataFilename, textFilename, matFilename, numIterations, eta):
    """
    Computes the inversion method fitted to a diffusion model to estimate
    absorption and reduced scattering coefficients from reflectance profiles
    data obtained through the Dia-Stron TLS850.
    The annotation Method 2 refers to performing the inversion method on an
    average of all records, thus obtaining optical properties for it. It is
    faster than the Method 1, but does not allow running statistical studies.

    Args:
    - dataFilename (str): filename of .txt file containing the reflectance
    profiles obtained by acquisition through the TLS device.
    - textFilename (str): filename of .txt file to save the final estimates of
    absorption and reduced scattering coefficients for each RGB channel.
    - matFilename (str): filename of .mat file to access variables finalAbs and
    finalSca containing the final values of absorption and reduced scattering
    coefficients. This corresponds to the average of optical estimates.
    - numIterations (int): integer to specify the number of iterations in the
    convergence loop to reach final values of estimates.
    - eta (float): real part of the refractive index of the material.
    """
    # Creating text file for the results
    resultsID = open(textFilename, 'w')

    # Reading data from TLS text file that must be exported from the software
    # first. All records for each channel are extracted and stored in the
    # following variables.
    dataR, dataG, dataB = read_data_TLS(dataFilename)
    numRecords = dataR.shape[1]

    # Diffuse Fresnel parameter varying depending on the material and the real
    # part of the refractive index
    A = get_diffuse_fresnel(eta)

    # Modelisation of the NMOS sensor array. Its length is of 20 millimeters
    # and it is needed to compute the diffusion model and obtain estimates of
    # optical coefficients in the proper unit.
    r = np.linspace(0, 20, 512)

    # Part 1: Initialization
    # Averaging all records
    avgDataR = np.mean(dataR, axis=1)
    avgDataG = np.mean(dataG, axis=1)
    avgDataB = np.mean(dataB, axis=1)

    # Initial guesses
    eff0 = np.zeros(3)
    trp0 = np.zeros(3)

    # RED Channel
    proDataR, proDistanceR = get_preprocessed_data(avgDataR, r)
    sepR = get_threshold_index(proDistanceR, proDataR)
    trp, eff = compute_channel_fitting_init(A, proDistanceR, sepR, proDataR)
    trp0[0], eff0[0] = trp, eff

    # GREEN Channel
    proDataG, proDistanceG = get_preprocessed_data(avgDataG, r)
    sepG = get_threshold_index(proDistanceG, proDataG)
    trp, eff = compute_channel_fitting_init(A, proDistanceG, sepG, proDataG)
    trp0[1], eff0[1] = trp, eff

    # BLUE Channel
    proDataB, proDistanceB = get_preprocessed_data(avgDataB, r)
    sepB = get_threshold_index(proDistanceB, proDataB)
    trp, eff = compute_channel_fitting_init(A, proDistanceB, sepB, proDataB)
    trp0[2], eff0[2] = trp, eff

    # Part 2: Iteration process for convergence
    # Storing data for each iterations
    saveEff = np.zeros((1, 3, numIterations + 1))
    saveTrp = np.zeros((1, 3, numIterations + 1))

    # Initialization of saving matrices
    saveEff[0, :, 0] = eff0
    saveTrp[0, :, 0] = trp0

    # Iteration loop
    for i in range(1, numIterations + 1):
        print("Iteration #", i)

        eff = saveEff[0, :, i - 1]
        trp = saveTrp[0, :, i - 1]

        # Initialization
        effInt = np.zeros(3)
        trpInt = np.zeros(3)

        # RED Channel
        proDataR, proDistanceR = get_preprocessed_data(avgDataR, r)
        sepR = get_threshold_index(proDistanceR, proDataR)
        transport, efficient = compute_channel_fitting_loop(A, proDistanceR, sepR, proDataR, eff[0], trp[0])
        effInt[0], trpInt[0] = efficient, transport

        # GREEN Channel
        proDataG, proDistanceG = get_preprocessed_data(avgDataG, r)
        sepG = get_threshold_index(proDistanceG, proDataG)
        transport, efficient = compute_channel_fitting_loop(A, proDistanceG, sepG, proDataG, eff[1], trp[1])
        effInt[1], trpInt[1] = efficient, transport

        # BLUE Channel
        proDataB, proDistanceB = get_preprocessed_data(avgDataB, r)
        sepB = get_threshold_index(proDistanceB, proDataB)
        transport, efficient = compute_channel_fitting_loop(A, proDistanceB, sepB, proDataB, eff[2], trp[2])
        effInt[2], trpInt[2] = efficient, transport

        # Saving results for next iteration
        saveEff[0, :, i] = effInt
        saveTrp[0, :, i] = trpInt

    # Converting from (transport, efficient) to (absorption, reduced scattering)
    finalEff = saveEff[0, :, -1]
    finalTrp = saveTrp[0, :, -1]
    finalSca, finalAbs = get_optical_properties(finalTrp, finalEff)

    # Writing processes in the .txt file and .mat file
    with open(textFilename, 'w') as resultsID:
        resultsID.write(f'Abs (RGB): {finalAbs[0]:.3e} {finalAbs[1]:.3e} {finalAbs[2]:.3e}\n')
        resultsID.write(f'Sca (RGB): {finalSca[0]:.3e} {finalSca[1]:.3e} {finalSca[2]:.3e}\n')

    # Saving to .mat file
    scipy.io.savemat(matFilename, {'finalAbs': finalAbs, 'finalSca': finalSca})
