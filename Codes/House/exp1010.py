import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("Housing.csv")
prices  = np.array(data['price']/10000)
area = np.array(data['area'])
bedrooms = np.array(data['bedrooms'])



# ax = plt.axes(projection="3d")
plt.scatter(area,prices)
plt.show()