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
features = ['xcoord50', 'ycoord50', 'zcoord50', 'RV', 'pmDE', 'pmRA', 'Uvel50', 'Vvel50', 'Wvel50', 'Plx']
# Attempt to clean up the data
stars = stars.dropna()[1:]
for feature in features:
    stars[feature] = pd.to_numeric(stars[feature])
fig, ax = plt.subplots()

# sin, but in degrees
def sin(d):
    return np.sin(np.degrees(d))
# cos, but in degrees
def cos(d):
    return np.cos(np.degrees(d))
# Converts velocities (which are in the equatorial coordinate system) to the galactic coordinates used for position
# I wrote this function before I realized that these were pre-calculated already
def toGalacticCoordinates(RV, pmRA, pmDE, x, y, z):
    # rotation matrix obtained from
    # https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html
    rotation_matrix = [[-0.054876, -0.873437, -0.483835],
                       [0.494109, -0.444830, 0.746982],
                       [-0.867666, -0.198076, 0.455984]]
    rotation_matrix = np.array(rotation_matrix)
    p_ICRS = np.array([-sin(pmRA), cos(pmDE), 0])
    p_ICRS = np.transpose(p_ICRS)
    q_ICRS = np.array([-cos(pmRA) * sin(pmDE), -sin(pmRA) * sin(pmDE), cos(pmDE)])
    q_ICRS = np.transpose(q_ICRS)
    pmRA = np.reshape(pmRA, [-1, 1])
    pmDE = np.reshape(pmDE, [-1, 1])
    pm_ICRS = p_ICRS * pmRA + q_ICRS * pmDE

    pm_galaxy = np.matmul(rotation_matrix, np.transpose(pm_ICRS))
    # The factor needed to convert from km/s to parsecs per 1 thousand years
    rvFactor = 0.001022012
    rv_galaxy = (np.array([x, y, z]) / np.linalg.norm([x, y, z])) * RV
    rv_galaxy *= rvFactor
    rv_galaxy = np.reshape(rv_galaxy, [-1, 1])
    # 1 arcsec/year to parsecs/1000 years at a distance of 1 parsec
    pmFactor = 0.000004848
    pm_galaxy *= pmFactor
    pm_galaxy *= np.sqrt(np.square(x) + np.square(y) + np.square(z))
    total_velocity = pm_galaxy + rv_galaxy
    return total_velocity
stars['total_velocity'] = stars.apply(lambda x: np.sqrt(np.sum(np.square([x['Uvel50'], x['Vvel50'], x['Wvel50']]))), axis=1)
# Further stars will be rendered in front of closer ones in matplotlib
stars = stars.sort_values('Plx', axis=0, ascending=True)

backtracked = stars.copy()
# If a star is traveling at x km/s then in 977,812 years it will have travelled x parsecs.
time_constant = 977812

kde = KernelDensity(bandwidth=5, kernel='gaussian')
kde.fit(np.vstack([backtracked['xcoord50'], backtracked['ycoord50'], backtracked['zcoord50']]).T)
years_backtracked = time_constant * 50
backtracked['xcoord50'] = stars['xcoord50'] - (years_backtracked / time_constant) * stars['Uvel50']
backtracked['ycoord50'] = stars['ycoord50'] - (years_backtracked / time_constant) * stars['Vvel50']
backtracked['zcoord50'] = stars['zcoord50'] - (years_backtracked / time_constant) * stars['Wvel50']
c = np.exp(kde.score_samples(np.vstack([stars['xcoord50'], stars['ycoord50'], stars['zcoord50']]).T))
print(np.mean(c))
print(np.std(c))
ax.scatter(backtracked['xcoord50'], backtracked['ycoord50'], c=c, cmap='coolwarm', norm="linear")
plt.show()