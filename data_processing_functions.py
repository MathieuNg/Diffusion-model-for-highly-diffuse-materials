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
import pandas as pd
from scipy.signal import find_peaks

def get_array_separated(data, sep):
    """
    Separates the array into two parts following the separator parameter provided.

    Inputs:
    data (array-like): vector to be separated into two parts.
    sep (int): separator position to use to separate the data array.

    Outputs:
    data1 (array-like): vector containing values before and including the separator.
    data2 (array-like): vector containing values after and excluding the separator.
    """
    data1 = data[:sep]
    data2 = data[sep + 1:]

    return data1, data2

def get_diffuse_fresnel(eta):
    """
    Computes the diffuse Fresnel parameter to take into consideration
    internal reflections inside the material following the method proposed by
    Donner et al., "Light diffusion in multi-layered translucent materials,”
    ACM Trans. Graph. 24, 1032–1039 (2005).

    Inputs:
    eta (float): real part of the refractive index of the material considered.

    Outputs:
    A (float): diffuse Fresnel parameter defined by the following equation.
    """
    if eta < 1:
        Fd = -0.4399 + 0.7099 / eta - 0.3319 / eta ** 2 + 0.0636 / eta ** 2
    else:
        Fd = -1.440 / eta ** 2 + 0.710 / eta + 0.668 + 0.0636 * eta

    A = (1 + Fd) / (1 - Fd)
    return A

def get_optical_properties(transport, efficient):
    """
    Returns the values of absorption and reduced scattering coefficients by
    using the estimates of transport and efficient coefficients.

    Inputs:
    transport (float): transport coefficient given in mm-1.
    efficient (float): efficient coefficient given in mm-1.

    Outputs:
    scattering (float): reduced scattering coefficient given in mm-1.
    absorption (float): absorption coefficient given in mm-1.
    """
    absorption = efficient ** 2 / (3 * transport)
    scattering = transport - absorption

    return scattering, absorption

def get_preprocessed_data(data, distance):
    """
    Preprocessing of the data. The reflectance profile cannot be negative so
    it removes potential negative values by setting them at 0. Also, to avoid
    noise from the detector in the first arrays, it is removing values before
    the peak of the maximum. Most of the time, it is cutting the first two
    arrays of the NMOS sensor.

    Inputs:
    data (array-like): vector of the data and selected record to preprocess.
    distance (array-like): vector of the radial distance corresponding to the data.
    Since data and distance must have the same dimension, it needs to be
    adapted if changes occur.

    Outputs:
    newData (array-like): vector of the new data to be used for inversion.
    newDistance (array-like): vector of the new radial distance.
    """
    # Setting negative values to zero
    data[data < 0] = 0

    # Cutting values before maximum
    max_index = np.argmax(data)
    new_data = data[max_index:]
    new_distance = distance[max_index:]

    return new_data, new_distance

def get_threshold_index(r, data):
    """
    Finds the distance for which the reflectance profile can be divided into
    two parts, one to estimate the efficient coefficient and the other for
    the transport coefficient. It follows the method described by Farrell et
    al., “The use of a neural network to determine tissue optical properties
    from spatially resolved diffuse reflectance measurements,”
    Phys. Med. & Biol. 37, 2281 (1992).

    Inputs:
    r (array-like): vector of the radial distance.
    data (array-like): vector containing the reflectance data profile of the
    corresponding channel and record.

    Output:
    index (int): integer number to be used as the position to separate the
    reflectance profile vector.
    """
    # Smoothing data with moving average filter
    dataS = np.convolve(r**2 * data / np.max(data), np.ones(10) / 10, mode='same')

    # Gradient of the smoothed data. We search for changes in the gradient.
    grad = np.gradient(dataS)

    # Find negative peaks
    peaks, _ = find_peaks(-grad, distance=10)

    # Look for the first set of peaks that satisfy the condition
    for peak in peaks:
        if np.sum(grad[peak:peak + 10] < 0) >= 6:
            return peak

    raise ValueError("Threshold index not found")

def read_data_TLS(filename):
    """
    Reads the text file generated by TLS software containing the reflectance
    data profiles and puts them in matrices depending on the RGB channels.

    Inputs:
    filename (str): indicates the path of the filename to read the data from.
    It should be a text file generated by the TLS software.

    Outputs:
    dataR (numpy.ndarray): matrix of shape (512 x numRecords) containing information of
    the RED channel.
    dataG (numpy.ndarray): matrix of shape (512 x numRecords) containing information of
    the GREEN channel.
    dataB (numpy.ndarray): matrix of shape (512 x numRecords) containing information of
    the BLUE channel.
    """
    # Read the table from the file
    file_table = pd.read_table(filename, skiprows=7, header=None)

    # Extract relevant information
    tableRecords = file_table.iloc[:,0]
    numRecords = int(tableRecords.iloc[-1])
    numChannels = int(float(file_table.iloc[-1,1]))
    tableData = file_table.iloc[:, 2]

    data = np.zeros((numChannels, numRecords))

    # Populate data matrix
    for i in range(numRecords):
        cond = tableRecords == str(i + 1)
        data[:, i] = tableData[cond].array

    # Separation of data from RGB channels
    dataR = data[:, ::3]
    dataG = data[:, 1::3]
    dataB = data[:, 2::3]

    return dataR, dataG, dataB
