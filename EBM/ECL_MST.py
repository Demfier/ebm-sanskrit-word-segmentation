
# coding: utf-8

# In[38]:

#!/usr/bin/env python3
import numpy as np
import sys
from collections import defaultdict, namedtuple
from copy import deepcopy

# Arc is same as edge
Arc = namedtuple('Arc', ('tail', 'weight', 'head'))


def min_spanning_arborescence(arcs, source):
    good_arcs = []
    quotient_map = {arc.tail: arc.tail for arc in arcs}
    quotient_map[source] = source
    while True:
        min_arc_by_tail_rep = {}
        successor_rep = {}
        for arc in arcs:
            if arc.tail == source:
                continue
            tail_rep = quotient_map[arc.tail]
            head_rep = quotient_map[arc.head]
            if tail_rep == head_rep:
                continue
            if tail_rep not in min_arc_by_tail_rep or min_arc_by_tail_rep[tail_rep].weight > arc.weight:
                min_arc_by_tail_rep[tail_rep] = arc
                successor_rep[tail_rep] = head_rep
        cycle_reps = find_cycle(successor_rep, source)
        if cycle_reps is None:
            good_arcs.extend(min_arc_by_tail_rep.values())
            return spanning_arborescence(good_arcs, source)
        good_arcs.extend(min_arc_by_tail_rep[cycle_rep] for cycle_rep in cycle_reps)
        cycle_rep_set = set(cycle_reps)
        cycle_rep = cycle_rep_set.pop()
        quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}
        
def find_cycle(successor, source):
    visited = {source}
    for node in successor:
        cycle = []
        while node not in visited:
            visited.add(node)
            cycle.append(node)
            node = successor[node]
        if node in cycle:
            return cycle[cycle.index(node):]
    return None


def spanning_arborescence(arcs, source):
    arcs_by_head = defaultdict(list)
    for arc in arcs:
        if arc.tail == source:
            continue
        arcs_by_head[arc.head].append(arc)
    solution_arc_by_tail = {}
    stack = arcs_by_head[source]
    while stack:
        stack = sorted(stack)
        arc = stack.pop(0)
        if arc.tail in solution_arc_by_tail:
            continue
        solution_arc_by_tail[arc.tail] = arc
        stack.extend(arcs_by_head[arc.tail])
    return solution_arc_by_tail


def MST_ECL(nodelist, WScalarMat, conflicts_Dict1, source):
    pre_nodes = []
    nodes = []
    WScalarMat1 = deepcopy(WScalarMat)
    
    mst_nodes = defaultdict(lambda: [])
    mst_nodes_bool = np.array([False]*len(nodelist))
    mst_adj_graph = np.ndarray(WScalarMat.shape, np.bool)*False
    
    while(len(nodes)<len(nodelist)):
        i = int(np.argmin(WScalarMat1)/len(nodelist))
        j = np.argmin(WScalarMat1)%len(nodelist)
        if(i not in nodes and j not in nodes and i!=j and i not in conflicts_Dict1[j]):
            pre_nodes.extend([i,j])
            nodes.extend([i,j])
            for x in conflicts_Dict1[i]:
                if x not in nodes:
                    nodes.append(x)
            for x in conflicts_Dict1[j]:
                if x not in nodes:
                    nodes.append(x)
        elif(i not in nodes and j in pre_nodes and i!=j and i not in conflicts_Dict1[j]):
            pre_nodes.append(i)
            nodes.append(i)
            for x in conflicts_Dict1[i]:
                if x not in nodes:
                    nodes.append(x)
        elif(j not in nodes and i in pre_nodes and i!=j and j not in conflicts_Dict1[i]):
            pre_nodes.append(j)
            nodes.append(j)
            for x in conflicts_Dict1[j]:
                if x not in nodes:
                    nodes.append(x)
        WScalarMat1[i][j] = sys.maxsize
    
    pre_nodes.sort()
    for i in pre_nodes:
        mst_nodes_bool[i] = True
        mst_nodes[nodelist[i].chunk_id].append(nodelist[i])
    mst_nodes = dict(mst_nodes)
    
    # list of arcs(edges)
    list_arcs = []
    for i in range(len(nodelist)):
        for j in range(len(nodelist)):
            if i in pre_nodes and j in pre_nodes and WScalarMat[i][j] != 0.0:
                list_arcs.append(Arc(j,WScalarMat[i][j],i))

    Resultant_Arcs = min_spanning_arborescence(list_arcs,pre_nodes[0])
    #print(Resultant_Arcs)
    for i in Resultant_Arcs.values():
        #print(i.head,i.tail)
        mst_adj_graph[i.head][i.tail] = True
    
    return(mst_nodes_bool,mst_nodes,mst_adj_graph)

# Example_Graph_1 = [Arc(1, 9, 0), Arc(2, 10, 0), Arc(3, 9, 0), Arc(2, 20, 1), Arc(3, 3, 1), Arc(1,30, 2), Arc(3, 30, 2), Arc(2, 0, 3), Arc(1, 11, 3)]
# Example_Graph_2 = [Arc(1, 10, 0),Arc(2, 7, 0),Arc(1, 9, 2),Arc(4, 10, 1),Arc(3, 2, 2),Arc(3, 20, 4),Arc(4, 23, 2),Arc(5, 5, 2),Arc(3, 7, 5)]   
# print(min_spanning_arborescence(Example_Graph_2,0))

