# Clean up the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
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

model = LinearRegression()
X = np.array(stars['xcoord50']).reshape(-1, 1)
y = stars['Uvel50']
model.fit(X, y)
print(model.score(X, y))
print(model.coef_, model.intercept_)
x_graph = np.linspace(-8000, 8000)
y_graph = model.coef_[0] * x_graph + model.intercept_
ax.scatter(stars['xcoord50'], stars['Uvel50'])
ax.plot(x_graph, y_graph)
plt.show()