# File: EDF to CSV converter
# Description: This program converts .edf files to .csv files.
# Institution: University of Texas at Austin, Department of Biomedical Engineering
# Developer: Shao-Po (Shawn) Huang
# Team Members: Bryce Carr, Ajay Gadwal, Ethan Muyskens, Christian Schonhoeft

# Date Last Modified: 05/11/20

# This program uses numpy and mne.
# Source: https://stackoverflow.com/questions/52293033/how-to-convert-edf-file-into-csv-file-in-python

import numpy as np
import mne

# Convert .edf to .csv by providing a path to the original .edf file and the destination for the new .csv file
PATH = PATH_EDF
edf = mne.io.read_raw_edf(PATH)
header = ','.join(edf.ch_names)
PATH2 = PATH_CSV
np.savetxt(PATH2, edf.get_data().T, delimiter=',', header=header)

