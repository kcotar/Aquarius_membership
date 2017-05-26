from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import numpy as np

data = np.array([[1,2],[3,2],[7,5],[7,-5],[-5,-5],[-5,-5],[-5,-5],[-5,-5],[-5,-5],[-5,-5],[-5,-5],[-5,-5],[-5,-5]])

stars_density = KernelDensity(bandwidth=5, kernel='epanechnikov').fit(data)
grid_pc = 10
grid_pos = np.linspace(-grid_pc, grid_pc, 2000)
_x, _y = np.meshgrid(grid_pos, grid_pos)
print 'Computing density field'
density_field = stars_density.score_samples(np.vstack((_x.ravel(), _y.ravel())).T) + np.log(data.shape[0])
density_field = np.exp(density_field).reshape(_x.shape)

plt.imshow(density_field, interpolation=None, cmap='seismic', origin='lower', vmin=0., vmax=.2)
plt.colorbar()
plt.show()