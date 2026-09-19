import numpy as np
import matplotlib.pyplot as plt

print("NumPy version :", np.__version__)
print("Matplotlib version :", plt.matplotlib.__version__)

# Petit test
X = np.random.rand(10, 2)
print("Données générées :", X.shape)