# Clean up the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
# in the file, values that are all spaces are na
na_values = [" " * i for i in range(15)]
# Import data as pandas DataFrame
# This data was accessed from https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/A%2bA/649/A6/table1c
stars = pd.read_table("nearby_stars.tsv", delimiter=';', header=60, na_values=na_values)
features = ['BPmag', 'RPmag', 'Gmag']
# Attempt to clean up the data
stars = stars.dropna()[1:]
for feature in features:
    stars[feature] = pd.to_numeric(stars[feature])
fig, ax = plt.subplots()

# sin, but in degrees
stars['color_index'] = stars['BPmag'] - stars['Gmag']
ax.scatter(stars['color_index'], stars['Gmag'])
ax.set_y
plt.show()