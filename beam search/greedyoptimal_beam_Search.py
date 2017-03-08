import pickle
import os
import numpy as np
from random import randint
import random
input_path ='/home/prithvi/Downloads/extract/extras/all_dcs_lemmas_matrix.p'
input_load = pickle.load(open(input_path, "rb"))

class beamsearch:
    def __init__(self,k,levels,max_conflicting_nodes):
        self.k=k    #beam search width
        self.levels = levels   # number of levels
        self.max_conflicting_nodes = max_conflicting_nodes  #number of maximum conflictiong nodes in a level
        #generating a list of number of confliction nodes at each level
        self.cn_at_each_level=self.g(levels,max_conflicting_nodes)  
        #generating the nodes ie. the node indexes which are conflicting at each level.This is a list of lists
        self.nodes_list=self.generate_nodes(self.cn_at_each_level)
        print("List of node indexes at each level")
        print(self.nodes_list)
        print("Top k  paths greedy")
        self.k_paths_greedy=self.top_k_paths_greedy(self.nodes_list,k)
        print(self.k_paths_greedy)
        print("Top k  paths optimal")
        self.k_paths_optimal,self.p_values=self.top_k_paths_optimal(self.nodes_list,k)
        print(self.k_paths_optimal)
        print(self.p_values)
    def g(self,levels,m):
        return [randint(1,m) for x in range(0,levels)]
    def generate_nodes(self,c):
        return [random.sample(range(0, 66914), i) for i in c]
    def top_k_paths_greedy(self,l,k):
        s = [[] for _ in range(k)]
        for i in range(0,len(l)):
            if i==0:
                temp = []
#                 if(len(l[i])>k):
#                     v = random.sample(range(0,len(l[i])),k)
#                     for j in v:
#                         temp.append([l[i][j]])
#                 else:
                for j in range(0,len(l[i])):
                    temp.append([l[i][j]])
                s=temp
            else:
                count = 0
                for m in s:
                    if m:
                        count+=1
                t = np.zeros(shape=(count,len(l[i]))) 
                for p in range(0,t.shape[0]):
                    for q in range(0,t.shape[1]):
                        t[p,q]=input_load[s[p][-1]][l[i][q]]
                t_1d = t.flatten()
                idx_1d = t_1d.argsort()[-k:][::-1]
                x_idx, y_idx = np.unravel_index(idx_1d, t.shape)
                temp = []
                for x, y, in zip(x_idx, y_idx):
                    temp.append(s[x]+[l[i][y]])
                s = temp
        return s
    def top_k_paths_optimal(self,l,k):
        s = [[] for _ in range(k)]
        p_values=[]
        for i in range(0,len(l)):
            if i==0:
                temp = []
#                 if len(l[i])>k:
#                     v = random.sample(range(0,len(l[i])),k)
#                     for j in v:
#                         temp.append([l[i][j]])
#                 else:
                for j in range(0,len(l[i])):
                    temp.append([l[i][j]])
                p_values = [0.5]*len(temp)
                s=temp
            else:
                count = 0
                for m in s:
                    if m:
                        count+=1
                t = np.zeros(shape=(count,len(l[i]))) 
                for p in range(0,t.shape[0]):
                    for q in range(0,t.shape[1]):
                        t[p,q]=input_load[s[p][-1]][l[i][q]]
                t = (t.T * p_values).T
                t_1d = t.flatten()
                idx_1d = t_1d.argsort()[-k:][::-1]
                x_idx, y_idx = np.unravel_index(idx_1d, t.shape)
                temp = []
                for x, y, in zip(x_idx, y_idx):
                    temp.append(s[x]+[l[i][y]])
                p_values = [t[x][y] for x, y, in zip(x_idx, y_idx)]
                s = temp
        return s,p_values        
def main():
    k=2
    levels=4
    max_conflicting_nodes=3
    bs = beamsearch(k,levels,max_conflicting_nodes)
if __name__ == "__main__": main()   

