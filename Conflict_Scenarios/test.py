two_d_array = [["a", "b", "c"],
               ["d", "e", "f"],
               ["g", "h", "i"]]
sets = [set(lst) for lst in two_d_array]
n = len(sets)

overlaps = []
for i in range(n):
    for j in range(i + 1, n):
        intersection = sets[i].intersection(sets[j])
        if intersection:
            overlaps.append((i, j, intersection))

print(len(overlaps))
