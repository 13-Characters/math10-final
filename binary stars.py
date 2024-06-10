import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Parse input
# Input is from https://www.astro.keele.ac.uk/jkt/debcat/
# Note that "-9.9900" is considered NaN
input = pd.read_table("binaries.txt", na_values = -9.9900, sep='\s+')
color_mapping = {"Main Sequence": 0, "Giant": 1, "Unknown": 2}
# Each row represents a system of two stars, but for this project, we want to make a table where
# one row is one star, so we must merge data
binaries = pd.DataFrame()
binaries['logM'] = pd.concat([input['logM1'], input['logM2']], ignore_index=True)
binaries['logR'] = pd.concat([input['logR1'], input['logR2']], ignore_index=True)
binaries['logL'] = pd.concat([input['logL1'], input['logL2']], ignore_index=True)
binaries['SpT'] = pd.concat([input['SpT1'], input['SpT2']], ignore_index=True)
print(binaries)
def isMainSequence(classification):
    # If the "Yerkes luminosity class" is IV or V, then the star is considered main sequence
    main_sequence_markers = ['_IV', '_V']
    giant_markers = ['_I', '_II', '_III']
    if any([marker in classification for marker in main_sequence_markers]):
        return "Main Sequence"
    if any([marker in classification for marker in giant_markers]):
        return "Giant"
    return "Unknown"

binaries['c'] = binaries.apply(lambda x: isMainSequence(x['SpT']), axis=1)



# filter for only known classifications
binaries = binaries[binaries['c'] != "Unknown"]
X = binaries['logR']
y = binaries['c']
c = binaries['c']
c = [color_mapping[c] for c in c]

fig, ax = plt.subplots()
ax.scatter(X, y, c=c)
plt.show()
# Stellar classification gives us a way to tell if a star is "main sequence", which obeys a mass-radius relationship, or if it is a giant, which obeys a different relationship.

# lr_model = LinearRegression()
# lr_model.fit(binaries['logM1'])