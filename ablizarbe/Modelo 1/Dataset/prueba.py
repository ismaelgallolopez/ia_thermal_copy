import numpy as np

potencias = np.zeros((13,13))

potencias[6,3], potencias[6,9], potencias[3,6], potencias[9,6] = 1, 3, 2, 4

print(potencias)