import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


intercept = 40
gradient = 0.5 #slop

startx = 0 #start generating the data
endx = 200 # end of the data generation



midx = (endx + startx)/2
midy = intercept + gradient * midx

limit = 250

perpendicular_gradinet = - 1 / gradient
perpendicular_intercept = midy + midx / gradient

x = np.arange(startx, endx, 5)
y = intercept + (gradient * x) + np.random.normal(scale = 10, size = len(x))
X = np.append(x.reshape(-1,1), y.reshape(-1,1), axis= 1)

model = PCA()
component =  model.fit_transform(X)

print( component)

var_x = np.var(x)
var_y = np.var(y)
total_variance = var_x + var_y
print(var_x, var_y, total_variance)
print(var_x / total_variance,  var_y / total_variance)


print()
var_pc1 = np.var(component[:,0])
var_pc2 = np.var(component[:,1])
total_variance_pc = var_pc1 + var_pc2
print(var_pc1, var_pc2, total_variance)
print(var_pc1 / total_variance_pc,  var_pc2 / total_variance_pc)

print(model.explained_variance_ratio_)
exit()
fig, axes = plt.subplots(figsize = (16, 8), ncols = 2)

ax = axes[0]
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.scatter(midx, midy, s = 100, color = 'red')
ax.plot(x, intercept + gradient * x)
ax.plot(x, perpendicular_intercept + perpendicular_gradinet * x)
ax.scatter(x, y, c = x)


ax = axes[1]
ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)
ax.scatter(component[:,0], component[:,1])

plt.show()
