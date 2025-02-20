import pandas as pd

movie = pd.read_csv('C:\GitHub\Machine-Learning\Practice ML\INF1279\movie.csv')

#print(movie.sort_values('Revenue (Millions)', ascending=False).head(5))
#print(movie.loc[movie['Revenue (Millions)'].nlargest(5).index,:])

print()

#print(movie.loc[movie['Revenue (Millions)'].nlargest(5).index,:])

from sklearn.preprocessing import MinMaxScaler
movie['Revenue_norm'] = MinMaxScaler().fit(movie[['Revenue (Millions)']]).transform(movie[['Revenue (Millions)']])
movie['Votes_norm'] = MinMaxScaler().fit(movie[['Votes']]).transform(movie[['Votes']])
movie['Metascore_norm'] = MinMaxScaler().fit(movie[['Metascore']]).transform(movie[['Metascore']])

#print(movie.head())

from sklearn.metrics.pairwise import euclidean_distances
distanceMatrix = euclidean_distances(movie.loc[[0,1,2,3],['Revenue_norm', 'Votes_norm', 'Metascore_norm']])
distanceMatrix = pd.DataFrame(distanceMatrix, index=movie.loc[[0,1,2,3],].index, columns=movie.loc[[0,1,2,3],].index)
print(distanceMatrix)

print(distanceMatrix.loc[1,0]==distanceMatrix.loc[0,1])
print(distanceMatrix.loc[0,1])

import math
euclidean_dist_0_1 = math.sqrt(
    math.pow(movie.loc[1,['Revenue_norm']]-movie.loc[0,['Revenue_norm']],2) +
    math.pow(movie.loc[1,['Votes_norm']]-movie.loc[0,['Votes_norm']],2) +
    math.pow(movie.loc[1,['Metascore_norm']]-movie.loc[0,['Metascore_norm']],2)
)
print(euclidean_dist_0_1)