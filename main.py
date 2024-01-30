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


from computing_functions import compute_inversion_method1, compute_inversion_method2

# Data filename should be precised with the .txt extension, with the folder
# root included in the name.
dataFilename = '.txt';

# Output will give a .txt file and a .mat file. Text file contains the average estimates of
# absorption and reduced scattering coefficients for RGB channels given in
# mm-1. 
textFilename = '.txt';

# Matrix file contains two variables finalAbs and finalSca. Both are
# matrices with values of absorption and reduced scattering coefficients
# for each record (method 1) or for the average of records (method 2).
matFilename = '.mat';

# Creating a .mat file containing the final estimates of absorption and
# reduced scattering coefficients for all RGB channels and records.
# Variables stored in this matrix are statAbs and statSca.
statFilename = '.mat';

# Number of iterations to loop to have final estimates of the coefficients.
# Convergence is reached quite rapidly in most cases.
numIterations = 20;

# Real part of the refractive index
eta = ;

# METHOD 1: Averaging estimates
# print('### Method 1 ###\n');
# compute_inversion_method1(dataFilename, textFilename, matFilename, statFilename, numIterations, eta);

# METHOD 2: Averaging signals
# print('### Method 2 ###\n');
# compute_inversion_method2(dataFilename, textFilename, matFilename, numIterations, eta);
