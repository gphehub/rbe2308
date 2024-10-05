import numpy as np
import csv
import math

# Generating the initial circuit parameters:
param = np.random.rand(100) * math.pi
np.savetxt('data/parameters.csv', param, delimiter=",")
