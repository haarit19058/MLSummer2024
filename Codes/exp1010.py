import numpy as np
import matplotlib.pyplot as plt


x_val = 100 * np.random.rand(10)

y_val = 100 * np.random.rand(10)


x_mesh , y_mesh = np.meshgrid(x_val,y_val)

z_val = np.cos(x_mesh) + np.sin(y_mesh)

ax = plt.axes(projection = "3d")
ax.plot_surface(x_mesh,y_mesh,z_val,cmap="viridis")
plt.show()