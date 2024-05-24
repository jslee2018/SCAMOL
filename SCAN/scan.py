"""scan.py illustrates how to use the pregel.py library and test
    that model works
"""
from pregel import Vertex,Pregel
import threading

num_workers =  2
num_vertices =  14
threadLock = threading.Lock()

def main():

    # edges = [[6,4,5,1],[0,5,2],[1,3,5],[2,6,4,5],[3,6,0,5],
    #             [2,3,4,0,1],[3,4,0,10,11,7],[6,11,12,8],[7,12,9],
    #             [8,12,13,10],[6,9,11,12],[6,7,10,12],[11,10,7,8,9],
    #             [9]]
    filename = "data/TC1-1/1-1.dat"
    
    edges = []

    with open(filename) as file:
        for line in file:
            line = line.split()
            src = int(line[0].strip()) - 1
            dst = int(line[1].strip()) - 1
            
            if src + 1 > len(edges):
                edges.append([])
            edges[src].append(dst)
            
    
    vertice_ids  = range(len(edges))
    communities = []
    for i in range(len(edges)):
        v = ScanVertex(i, None, edges[i])
        communities.append(v)

    for i in range(len(edges)):
        neighbors = []
        for j in edges[i]:
            neighbors.append(communities[j])
        communities[i].neighbors = neighbors
            

    #initialize
    communities[12].value = 'A'
    communities[5].value = 'B'
    print(id(communities))
    '''for v in vertices:
        print(v.id,v.value)'''
    print('running scan pregel')
    p = Pregel(communities,num_workers)
    p.run()
    print(id(communities))
    for v in communities:
        print(v.id,v.value)
    

class ScanVertex(Vertex):
    
    def update(self):
          if self.superstep < 2:
            
              for v,m in self.incoming_messages:
                  if m is not None:
                      self.value = m
                          
              self.outgoing_messages = []
              for v in self.neighbors:
                 self.outgoing_messages.append((v,self.value))
                         
          else:
              self.active = False

if __name__ == "__main__":
    main()
