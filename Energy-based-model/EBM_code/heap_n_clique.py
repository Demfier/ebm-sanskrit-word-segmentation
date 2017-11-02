import numpy as np
from word_definite import *
import math

def Parent(i):
    return max(0, math.floor((i - 1)/2))

def Left(i):
    return 2*i + 1

def Right(i):
    return 2*(i + 1)


"""
################################################################################################
######################## NOMINAL NODE CLASS REQUIRED FOR USING  ################################
#########################  WITH THE HEAP DATA STRUCTURE  #######################################
################################################################################################
"""

class Node:
    def __init__(self, id, dist):
        self.dist = dist
        self.id = id
        self.isConflicted = False
        self.src = -1

"""
################################################################################################
############################  IMPLEMENTATION OF HEAP  ##########################################
################################################################################################
"""

class Heap:
    # It's a minHeap
    # Nodes are of type Word_definite
    def __init__(self, nodeList):
        self.nodeList = [n for n in nodeList]
        self.len = len(nodeList)
        self.idLocator = {}
        for i in range(self.len):
            self.idLocator[nodeList[i].id] = i
        self.Build()
            
    def Exchange(self, i, j):
        t = self.nodeList[i]
        self.nodeList[i] = self.nodeList[j]
        self.nodeList[j] = t
        self.idLocator[self.nodeList[i].id] = i
        self.idLocator[self.nodeList[j].id] = j
        
    def Decrease_Key(self, node, newDist, src):
        if node.isConflicted:
            return       
        i = self.idLocator[node.id]
        if newDist > node.dist:
            # relaxation not possible
            return
        else:
            node.dist = newDist
            node.src = src
            parent = Parent(i)
            while ((i > 0) and (self.nodeList[parent].dist > self.nodeList[i].dist)):
                self.Exchange(i, parent)
                i = parent
                parent = Parent(i)
                
    def Pop(self):
        if(self.len == 0):
            return None
        if(self.nodeList[0].isConflicted):
            # print("Pop has seen conflict!!!")
            return None
        
        # Remove the entry from the top of the heap
        nMin = self.nodeList[0]
        self.idLocator[self.nodeList[0].id] = -1
        
        # Put the last node on top of heap and heapify
        self.nodeList[0] = self.nodeList[self.len - 1]
        self.idLocator[self.nodeList[0].id] = 0
        self.len -= 1
        self.Min_Heapify(0)
        return nMin
        
    def Min_Heapify(self, i):
        nMin = self.nodeList[i]
        li = Left(i)
        if(li < self.len):
            if(self.nodeList[li].dist < nMin.dist):
                nMin = self.nodeList[li]
                min_i = li                
        ri = Right(i)
        if(ri < self.len):
            if(self.nodeList[ri].dist < nMin.dist):
                nMin = self.nodeList[ri]
                min_i = ri                
        if(nMin.id != self.nodeList[i].id):
            self.Exchange(i, min_i)
            self.Min_Heapify(min_i)
            
    def Delete(self, node):
        i = self.idLocator[node.id]
        self.nodeList[i].isConflicted = True
        self.nodeList[i].dist = np.inf
        self.Min_Heapify(i)
        
    def Build(self):
        self.len = len(self.nodeList)
        for i in range(int(Parent(self.len - 1)) + 1):
            self.Min_Heapify(i)
    
    def Print(self):
        i = 0
        level = 1
        ilimit = 0
        while(i < self.len):
            print('N(%d, %2.1f)' % (self.nodeList[i].id, self.nodeList[i].dist), end = ' ')
            i += 1
            if(i > ilimit):
                print('\n')
                level *= 2
                ilimit += level

"""
################################################################################################
######################  IMPLEMENTATION OF PRIM'S ALGO FOR FINDING MST ##########################
#############################  USES HEAP DEFINED ABOVE  ########################################
################################################################################################
"""
def MST(nodelist, WScalarMat, conflicts_Dict, source):
    # WTF Dude!!! This function should not be used... It is running Prim's on a directed graph!!!
    # Doesn't return MST
    mst_adj_graph = np.ndarray(WScalarMat.shape, np.bool)*False
    # print(len(nodelist))
    # Reset nodes and put ids
    for id in range(len(nodelist)):
        nodelist[id].id = id
        nodelist[id].dist = np.inf
        nodelist[id].isConflicted = False
        nodelist[id].src = -1
        
    # Initialize Graph and min-Heap
    nodelist[source].dist = 0
    for neighbour in range(len(nodelist)):
        if neighbour != source:
            nodelist[neighbour].dist = WScalarMat[source][neighbour]
            nodelist[neighbour].src = source
    h = Heap(nodelist)
    
    mst_nodes = defaultdict(lambda: [])
    mst_nodes_bool = np.array([False]*len(nodelist))
    # Run MST only until first conflicting node is seen
    # Conflicting node will have np.inf as dist
    while True:
        nextNode = h.Pop()
        if nextNode == None:
            break
        print("next-id:"+str(nextNode.id))
        print('picked by '+str(nodelist[nextNode.id].dist))
        print()
        # print(nextNode.src, nextNode.id, nextNode)
        mst_nodes_bool[nextNode.id] = True
        mst_nodes[nextNode.chunk_id].append(nextNode)


        if nextNode.src != -1:
            mst_adj_graph[nextNode.src, nextNode.id] = True
            # mst_adj_graph[nextNode.id, nextNode.src] = True
        nid = nextNode.id
        for conId in conflicts_Dict[nid]:
            h.Delete(nodelist[conId])
        for neighbour in range(len(nodelist)):
            if neighbour != nextNode.id:
                print(WScalarMat[nextNode.id][neighbour])
                print(nodelist[neighbour].dist)    
                h.Decrease_Key(nodelist[neighbour], WScalarMat[nextNode.id][neighbour], nextNode.id)

        print(mst_nodes_bool)
        # print(mst_nodes_bool)
    print('#'*30)
    mst_nodes = dict(mst_nodes)

    return (mst_nodes, mst_adj_graph, mst_nodes_bool)

def clique(nodelist, WScalarMat, conflicts_Dict, source):
    # WTF Dude!!! This function should not be used... It is running Prim's on a directed graph!!!
    # Doesn't return MST
    mst_adj_graph = np.ndarray(WScalarMat.shape, np.bool)*False
    # print(len(nodelist))
    # Reset nodes and put ids
    # print('node-ids')
    for id in range(len(nodelist)):
        # print(id)
        nodelist[id].id = id
        nodelist[id].dist = np.inf
        nodelist[id].isConflicted = False
        nodelist[id].src = -1
    # print('*'*40)
    # Initialize Graph and min-Heap
    nodelist[source].dist = 0

    nodeset=set()
    for neighbour in range(len(nodelist)):
        if neighbour != source:
            nodelist[neighbour].dist = WScalarMat[source][neighbour]
            nodelist[neighbour].src = source
            # nodeset.add((nodelist[neighbour].dist,neighbour))

    # nodeset = sorted(nodeset)
    nodeset.add((0,source))
    nodesadded=[]
    nodesavailable = np.zeros(len(nodelist),dtype=int)  # o if available, 1 if not available
    
    mst_nodes = defaultdict(lambda: [])
    mst_nodes_bool = np.array([False]*len(nodelist))
    # Run MST only until first conflicting node is seen
    # Conflicting node will have np.inf as dist

    it=0
    nextNode=-1
    while True:
        # print(nodeset)
        it+=1
        # print(it)
        if(it>1000):
            break
        if(len(nodeset)==0):
            break
        # print('before nn assign: ')
        # print(nextNode)
        nextNode = next(iter(nodeset))
        # print("after nn assign:")
        # print(nextNode)
        # print("Nextnode is :"+str(nextNode[1])+" Picked by :"+str(nextNode[0]))
        nextNode=nodelist[nextNode[1]]
        # print(type(nextNode))
        # print(st_setr(nextNode.id)+"",)

        # print(nextNode.id)
        nodesavailable[nextNode.id]=1
        # nodesavailable=1
        if nextNode == None:
            break
        # print(nextNode.src, nextNode.id, nextNode)
        mst_nodes_bool[nextNode.id] = True
        mst_nodes[nextNode.chunk_id].append(nextNode)

        nodeset = set()

        if nextNode.src != -1:
            mst_adj_graph[nextNode.src, nextNode.id] = True
            # mst_adj_graph[nextNode.id, nextNode.src] = True
        
        nid = nextNode.id
        nodesadded.append(nid)
        for conId in conflicts_Dict[nid]:
            # h.Delete(nodelist[conId])
            nodesavailable[conId]=1
            # print('here')
     
        for neighbour in range(len(nodelist)):
            # print(type(nodesavailable))
            # print(type(nodesavailable[0]))
            if(nodesavailable[neighbour]==1):
                continue
            if neighbour != nextNode.id:    
                # h.Decrease_Key(nodelist[neighbour], WScalarMat[nextNode.id][neighbour], nextNode.id)
                edgewt=0
                # print(nodesadded)
                for nodepresent in nodesadded:
                    edgewt+=WScalarMat[nodepresent][neighbour]
                # print('adding '+str(neighbour))
                nodeset.add((edgewt,neighbour))
                # print(nodeset)

        nodeset=sorted(nodeset)
        # print(nodeset)
    # print("#"*30)
    # print(mst_nodes_bool)
    # print('-'*20)
    # print('#'*30)
    mst_nodes = dict(mst_nodes)
    if(it>1000):
        print('!!!!*10')
    for i in range(len(mst_nodes_bool)):
        for j in range(len(mst_nodes_bool)):
            if(i==j):
                continue
            if(mst_nodes_bool[i] and mst_nodes_bool[j]):
                mst_adj_graph[i][j]=True
                mst_adj_graph[j][i]=True

    # print(mst_adj_graph)    
    # print("#")
    return (mst_nodes, mst_adj_graph, mst_nodes_bool)

def RandomST_GoldOnly(nodelist, WScalarMat, conflicts_Dict, source):
    (mst_nodes, mst_adj_graph, mst_nodes_bool) = MST(nodelist, WScalarMat, conflicts_Dict, source)

    mst_adj_graph = np.zeros_like(mst_adj_graph)
    nodelen = len(nodelist)
    
    ## Random mst_adj_graph
    free_set = list(range(nodelen))
    full_set = list(range(nodelen))
    st_set = []
    start_node = np.random.randint(nodelen)
    st_set.append(start_node)
    free_set.remove(start_node)
    for x in range(nodelen - 1):
        a = st_set[np.random.randint(len(st_set))]
        b = free_set[np.random.randint(len(free_set))]        
        if b not in st_set:
            st_set.append(b)
            free_set.remove(b)
        mst_adj_graph[a, b] = 1
        # mst_adj_graph[b, a] = 1 # Directed Spanning tree
    
    return (mst_nodes, mst_adj_graph, mst_nodes_bool)


def GetMSTWeight(mst_adj_graph, WScalarMat):
    return np.sum(WScalarMat[mst_adj_graph])
