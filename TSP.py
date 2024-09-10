#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 22:40:14 2021

@author: chenyuchun
"""
import time
import numpy as np
from sys import maxsize
from itertools import combinations
import copy
import random as rd
from matplotlib import pyplot as plt


def createMatrix(n):   #隨機產生一個矩陣
    M = np.random.randint(1,31, size=(n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] != M[j][i]:
                M[i][j] = M[j][i]
    return M
 
#TSP
#Dynamic Programming
def travel(n, W, P):
    D = {}    
    for i in range(n-1):
        D[i+1, ()] = W[i+1][0]
    
    temp_D1 = {}
    for k in range(1, n-1):
        comb = combinations(A, k)
        for x in comb:
            for i in range(1, n):          
                if i not in list(x) :  #vi is in A
                    for j in list(x):
                        K = list(set(x)-{j})
                        K.sort()
                        temp_D1[i, tuple(x), j] = W[i][j] + D[j, tuple(K)]
                    D[i, tuple(x)] = maxsize
                    for j in list(x):   #get minimum
                        if temp_D1[i, tuple(x), j] < D[i, tuple(x)]:
                            D[i, tuple(x)] = temp_D1[i, tuple(x), j]
                            P[i, tuple(x)] = {j}  
    D[0, tuple(A)] = maxsize
    for j in range(1, n):
        K = list(A-{j})
        temp_D1[0, tuple(A), j] = W[0][j] + D[j, tuple(K)]
    for j in range(1, n):   #get minimum
        if temp_D1[0, tuple(A), j] < D[0, tuple(A)]:
            D[0, tuple(A)] = temp_D1[0, tuple(A), j]
            P[0, tuple(A)] = {j}
    min_length = D[0, tuple(A)]
    return D,P, min_length

def getPath(n, P):
    path = []
    k = {0}
    B = V
    path.append(list(k)[0])
    for i in range(n-1):
        B = B - k
        temp = list(B)
        temp.sort()
        k = P[list(k)[0], tuple(temp)]
        path.append(list(k)[0])
    path.append(0)
    return path

#Genetic Algorithm
class Location:
    def __init__(self, x):
        self.loc = x

    def distance_between(self, location2):
        assert isinstance(location2, Location)
        return W[self.loc][location2.loc]


def create_locations(n):
    locations =[]
    x = np.arange(n).tolist()
    for x in x:
        locations.append(Location(x))
    return locations

class Route:
    def __init__(self, path):
        # path is a list of Location obj
        self.path = path
        self.length = self._set_length()

    def _set_length(self):
        total_length = 0
        path_copy = self.path[:]
        from_here = path_copy.pop(0)
        init_node = copy.deepcopy(from_here)
        while path_copy:
            to_there = path_copy.pop(0)
            total_length += to_there.distance_between(from_here)
            from_here = copy.deepcopy(to_there)
        total_length += from_here.distance_between(init_node)
        return total_length


class GeneticAlgo:
    def __init__(self, locs, level=10, populations=100, variant=3, mutate_percent=0.1, elite_save_percent=0.1):
        self.locs = locs
        self.level = level
        self.variant = variant
        self.populations = populations
        self.mutates = int(populations * mutate_percent)
        self.elite = int(populations * elite_save_percent)

    def _find_path(self):
        # locs is a list containing all the Location obj
        locs_copy = self.locs[:]
        path = []
        while locs_copy:
            to_there = locs_copy.pop(locs_copy.index(rd.choice(locs_copy)))
            path.append(to_there)
        return path

    def _init_routes(self):
        routes = []
        for _ in range(self.populations):
            path = self._find_path()
            routes.append(Route(path))
        return routes

    def _get_next_route(self, routes):
        routes.sort(key=lambda x: x.length, reverse=False)
        elites = routes[:self.elite][:]
        crossovers = self._crossover(elites)
        return crossovers[:] + elites

    def _crossover(self, elites):
        # Route is a class type
        normal_breeds = []
        mutate_ones = []
        for _ in range(self.populations - self.mutates):
            father, mother = rd.choices(elites[:4], k=2)
            index_start = rd.randrange(0, len(father.path) - self.variant - 1)
            # list of Location obj
            father_gene = father.path[index_start: index_start + self.variant]
            father_gene_names = [loc.loc for loc in father_gene]
            mother_gene = [gene for gene in mother.path if gene.loc not in father_gene_names]
            mother_gene_cut = rd.randrange(1, len(mother_gene))
            # create new route path
            next_route_path = mother_gene[:mother_gene_cut] + father_gene + mother_gene[mother_gene_cut:]
            next_route = Route(next_route_path)
            # add Route obj to normal_breeds
            normal_breeds.append(next_route)

            # for mutate purpose
            copy_father = copy.deepcopy(father)
            idx = range(len(copy_father.path))
            gene1, gene2 = rd.sample(idx, 2)
            copy_father.path[gene1], copy_father.path[gene2] = copy_father.path[gene2], copy_father.path[gene1]
            mutate_ones.append(copy_father)
        mutate_breeds = rd.choices(mutate_ones, k=self.mutates)
        return normal_breeds + mutate_breeds

    def evolution(self):
        routes = self._init_routes()
        for _ in range(self.level):
            routes = self._get_next_route(routes)
        routes.sort(key=lambda x: x.length)
        return routes[0].path, routes[0].length
  
    
t_total = time.time()

vertex = []
total_time_dp = []
total_time_ga = []
total_error = []
for num_vertex in range(4,21):
    print("number of vertex:", num_vertex)
    vertex.append(num_vertex)
    time_dp = 0
    time_ga = 0
    accumulative_error = 0
    for i in range(5):   
        W = createMatrix(num_vertex)
        # print("matrix:", W)
        V = set(range(0, num_vertex))
        A = V-{0}
        P = {}
        t_dp = time.time()
        print("DP:")
        D, P, min_length = travel(num_vertex, W, P)
        Path = getPath(num_vertex, P)
        # print("path:", Path)
        print("cost of minimum length {} :".format(i+1), min_length)
        runtime_dp = round(time.time() - t_dp, 5)
        print("runtime:", runtime_dp)
        time_dp += round(runtime_dp, 5)
        print("")
        
        t_ga = time.time()
        print("GA:")
        my_locs = create_locations(num_vertex)
        my_algo = GeneticAlgo(my_locs, level=40, populations=150, variant=2, mutate_percent=0.02, elite_save_percent=0.15)
        best_route, best_route_length = my_algo.evolution()
        best_route.append(best_route[0])
        print("cost of minimum length {} :".format(i+1), best_route_length)
        # print("path:", [loc.loc for loc in best_route])
        runtime_ga = round(time.time() - t_ga, 5)
        print("runtime:", runtime_ga)
        time_ga += round(runtime_ga, 5)
        print("")
        error = round(abs(best_route_length - min_length) / min_length, 5)
        accumulative_error += error
    avgtime_dp = round(time_dp / 5, 5)
    avgtime_ga = round(time_ga / 5, 5)
    avg_error = round(accumulative_error / 5, 5)
    total_time_dp.append(avgtime_dp)
    total_time_ga.append(avgtime_ga)
    total_error.append(avg_error)
print(total_time_dp)
print(total_time_ga)
print(total_error)
    
 
runtime_total = round(time.time() - t_total, 5)
print("Total runtime:", runtime_total)       

plt.figure(figsize=(7, 3))
plt.subplots_adjust(wspace=0.4)
plt.subplot(1, 2, 1)    
plt.plot(vertex, total_time_dp,linewidth=1.0, linestyle='--',label='DP')
plt.plot(vertex, total_time_ga,linewidth=1.0, linestyle='--',label='GA')
plt.scatter(vertex, total_time_dp)
plt.scatter(vertex, total_time_ga)
plt.legend(loc='upper left', shadow=True,prop={'size': 7}) 
plt.xlabel('Num of vertex')
plt.ylabel('avg.time')
    
plt.subplot(1, 2, 2) 
plt.plot(vertex, total_error)
plt.scatter(vertex, total_error)
plt.xlabel('Num of vertex')
plt.ylabel('avg.error')

