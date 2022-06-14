from ast import Not
from audioop import mul
from typing import List, Optional, Set, Tuple, TypeVar
from abc import ABC, abstractmethod
import networkx as nx
import matplotlib.pyplot as plt
from functools import reduce
import itertools

EdgeData = Tuple[int,int,float]
Nat = int

class PlanarNetwork(nx.DiGraph):
    def __init__(self, n:Nat, edge_list:List[List[EdgeData]]):
        super().__init__()
        for x in range(1,n+1):
            self.add_node(f"{x}_0")
        sink_number = {x:0 for x in range(1,n+1)}
        for stage in edge_list:
            already_incremented = []
            for edge in stage:
                from_node, to_node, weight = edge
                if from_node in already_incremented:
                    from_number = sink_number[from_node]-1
                else:
                    from_number = sink_number[from_node]
                if to_node in already_incremented:
                    to_number = sink_number[to_node]
                else:
                    to_number = sink_number[to_node]+1
                    sink_number[to_node]+=1
                    self.add_edge(f"{to_node}_{to_number-1}",f"{to_node}_{to_number}",weight=1)
                    already_incremented.append(to_node)
                self.add_node(f"{to_node}_{to_number}")
                self.add_edge(f"{from_node}_{from_number}",f"{to_node}_{to_number}",weight=weight)
        self.sink_number = sink_number
        
    def path_weight(self,path_nodes,multiplicative_identity=1.0):
        path_edges = [self[path_nodes[i]][path_nodes[i+1]] for i in range(len(path_nodes)-1)]
        return reduce(lambda x,y: x*y,[x['weight'] for x in path_edges],multiplicative_identity)

    def weight_matrix(self,i : Nat,j : Nat,additive_identity=0.0,multiplicative_identity=1.0):
        source_node = f"{i}_0"
        sink_node = f"{j}_{self.sink_number[j]}"
        weight = additive_identity
        for path_nodes in nx.all_simple_paths(self,source_node,sink_node):
            weight += self.path_weight(path_nodes,multiplicative_identity=multiplicative_identity)
        return weight
    
    def vertex_disjoint_collection(self,i_set : Set[Nat],j_set : Set[Nat], multiplicative_identity = 1.0, additive_identity = 0.0):
        for collection in pairings(i_set,j_set):
            path_collection_iterator = None
            for i,j in collection:
                ij_path = nx.all_simple_paths(self,f"{i}_0",f"{j}_{self.sink_number[j]}")
                if path_collection_iterator is None:
                    path_collection_iterator = ij_path
                else:
                    path_collection_iterator = itertools.product(path_collection_iterator,ij_path)
            # path_collection_iterator = itertools.product([nx.all_simple_paths(self,f"{i}_0",f"{j}_{self.sink_number[j]}") for i,j in collection])
            for path_collection in path_collection_iterator:
                seen_vertices = set({})
                is_vertex_disjoint = True
                my_paths = []
                for path in path_collection:
                    my_paths.append(list(path))
                    if seen_vertices.isdisjoint(set(path)):
                        seen_vertices = seen_vertices.union(set(path))
                    else:
                        is_vertex_disjoint = False
                        break
                if not is_vertex_disjoint:
                    # not vertex disjoint so contribute 0
                    yield additive_identity
                else:
                    all_path_weights = [self.path_weight(path_nodes,multiplicative_identity = multiplicative_identity) for path_nodes in my_paths]
                    yield reduce(lambda x,y: x*y,all_path_weights,multiplicative_identity)
    
    def lindstrom_minor(self,i_set : Set[Nat],j_set : Set[Nat], multiplicative_identity = 1.0, additive_identity = 0.0):
        minor_value = additive_identity
        for contrib in self.vertex_disjoint_collection(i_set,j_set, multiplicative_identity = multiplicative_identity, additive_identity = additive_identity):
            minor_value += contrib
        return minor_value


T = TypeVar
U = TypeVar
def pairings(i_set: Set[T],j_set : Set[U]):
    if not(len(i_set) == len(j_set)):
        raise ValueError("Must have same number of elements")
    if len(i_set)==0:
        yield set({})
        return
    xtemp = i_set.copy()
    i_popped = xtemp.pop()
    for i,j in {(i_popped,j) for j in j_set}:
        ytemp = j_set.copy()
        ytemp.remove(j)
        for remaining in pairings(xtemp,ytemp):
            remaining.add((i,j))
            yield remaining

if __name__ == "__main__":
    a = 1.2
    b = 2.2
    c = 3.2
    p = PlanarNetwork(3,[[(3,2,a),(3,3,1)],[(3,2,c),(2,1,b)]])
    for i in range(1,4):
        for j in range(1,4):
            print(f"a_({i},{j}) = {p.weight_matrix(i,j)}")
    from sympy import symbols
    A,B,C,D,E,F,G,H,I = symbols('a,b,c,d,e,f,g,h,i',commutative=True)
    ZERO, ONE = symbols('zero,one', commutative=True)
    p = PlanarNetwork(3,[[(3,2,A),(3,3,ONE)],[(3,2,C),(2,1,B)],[(2,2,E),(1,1,D)],[(3,3,F),(2,3,G),(1,2,H)],[(2,3,I)]])
    for i in range(1,4):
        for j in range(1,4):
            print(f"a_({i},{j}) = {p.weight_matrix(i,j,additive_identity=ZERO,multiplicative_identity=ONE)}")
    for g in pairings({1,2,3},{4,5,6}):
        print(g)
    print("Doing Lindstrom")
    for weight in p.vertex_disjoint_collection({2,3},{2,3},multiplicative_identity = ONE, additive_identity = ZERO):
        print(weight)
    print(p.lindstrom_minor({2,3},{2,3},multiplicative_identity = ONE, additive_identity = ZERO))