import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN



class WeirdClusters:

    def __init__(self):
        x = []
        y = []

        for i in np.arange(0, 2*pi, 0.01):
            r = np.random.uniform(90,100)
            x.append( r * cos(i))
            y.append( r * sin(i))

        for i in range(0,100, 1):
            x.append(np.random.normal(0,10))
            y.append(np.random.normal(0,10))



        self._x = np.array(x).reshape(-1,1)
        self._y = np.array(y).reshape(-1,1)
        self._X = np.append(self._x, self._y, axis = 1)

    

    def plot(self, clusters):

        fig = plt.figure(figsize= (9,9))
        ax = fig.add_subplot()
        ax.scatter(self._x, self._y, c = clusters, cmap = 'plasma')
        plt.show()
        

    def plot_neighbor_distances(self):
        nn = NearestNeighbors(n_neighbors= 4)
        nn.fit(self._X)

        # NB Scale / normalise the data is necessary First!!!

        distances, indices = nn.kneighbors()
        sorted_distances = np.sort(distances, axis = 0)
        print(distances)

        fig = plt.figure(figsize = (9,9))
        ax = fig.add_subplot()
        ax.set_xlabel("Sample number")
        ax.set_ylabel("distance to fruthest")
        ax.plot(sorted_distances[:,3])
        ax.axhline(y = 6,linestyle = "dashed")
        plt.show()

    def get_clusters(self):
        model = DBSCAN(eps = 7, min_samples = 4)  # radius of the circle = 6 # 2^2 = 4 therefore min_sample in each circle must be min 4 to call in core
        model.fit(self._X)
        return model.labels_
        
    
def main():
    w = WeirdClusters()
    #w.plot()
    w.plot_neighbor_distances()


    clusters = w.get_clusters()
    w.plot(clusters)
    


main()
        
    