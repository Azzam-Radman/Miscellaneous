from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(samples)

medians = np.zeros(tests.shape[0])

for i in range(tests.shape[0]):
    if i % 50000 == 0:
        print(i)
    row = tests[i, :].reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(row)
    distances, indices = nbrs.kneighbors(row)
    
    nearest_neighbors_idx = indices[np.argmin(distances.sum(axis=1))][1:]
    median = np.median(row[nearest_neighbors_idx])
    medians[i] += median
