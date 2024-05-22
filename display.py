import csv

filename = "data/TC1-6/1-6"

lines = []

clusters = {}
with open(f"{filename}.dat") as file:
    with open(f"{filename}-c.dat") as tfile:
        lines = file.readlines()
            
        for line in tfile:
            dot, cl = line.split()
            
            if cl not in clusters:
                clusters[int(cl)] = list()
                
            clusters[int(cl)].append(int(lines[int(dot)].split()[1]))
            
clusters_minmax = {}
for (cluster, l) in clusters.items():
    min = 1000000
    max = -1
    for i in l:
        if min > i:
            min = i
        if max < i:
            max = i
    
    clusters_minmax[cluster] = (min, max)
    
    
for (cluster, l) in sorted(list(clusters_minmax.items()), key=lambda x: x[1][1]):
    print(f"{cluster}: {l[0]}, {l[1]}")