from sklearn.metrics import normalized_mutual_info_score

# we need a score to compare these to lists (array) and say that they are the same (mutual_info)

target = [0, 0 , 0, 1, 1, 1]
#predicted = [3, 3, 3, 2, 2, 2]
predicted = [0, 1, 0, 1, 0, 1]

print(normalized_mutual_info_score(target, predicted))