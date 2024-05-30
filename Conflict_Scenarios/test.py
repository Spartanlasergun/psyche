tpc= range(2, 6, 1)
cs= range(10, 21, 1)
nb= range(5, 26, 1)
comp= [3]
umap_metric=['cosine']
hdb_metric=['euclidean']

# generate grid
grid = []
for a in tpc:
    for b in cs:
        for c in nb:
            for d in comp:
                for e in umap_metric:
                    for f in hdb_metric:
                        temp = [a, b, c, d, e, f]
                        grid.append(temp)