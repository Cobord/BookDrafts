from copy import deepcopy
from functools import reduce
from typing import Callable, Dict, List, Optional,Tuple, Union
import numpy as np

Nat = int
class Quiver:
    def __init__(self):
        self.vertices = set([])
        self.edges = []
        self.next_vertex_label = 0
        self.next_edge_label = 0
        self.edges_from = {}
        self.edges_to = {}
        self.edge_names = {}
        self.vertex_names = {}
    def add_vertex(self, name : str) -> Nat:
        self.vertices.add(self.next_vertex_label)
        self.vertex_names[self.next_vertex_label] = name
        self.next_vertex_label += 1
        return self.next_vertex_label-1
    def add_edge(self,source : Nat, target : Nat, edge_name : str) -> Nat:
        self.edges.append((source,target))
        self.edges_from[source] = self.edges_from.get(source,[]) + [self.next_edge_label]
        self.edges_to[target] = self.edges_to.get(target,[]) + [self.next_edge_label]
        self.edge_names[self.next_edge_label] = edge_name
        self.next_edge_label += 1
        return self.next_edge_label-1
    def from_edge_name(self,edge_name) -> Optional[Nat]:
        for (k,v) in self.edge_names.items():
            if v==edge_name:
                return k
        return None
    def __repr__(self):
        vertex_named_set = {self.vertex_names[z] for z in self.vertices}
        edges_with_names = [f"Edge {self.edge_names[i]} : {self.vertex_names[j]} -> {self.vertex_names[k]}" for (i,(j,k)) in enumerate(self.edges)]
        return f"Quiver with vertices {vertex_named_set} and edges {edges_with_names}"
    def __eq__(self,other):
        num_vertices_match = self.next_vertex_label == other.next_vertex_label
        num_edges_match = self.next_edge_label == other.next_edge_label
        if not(num_vertices_match and num_edges_match):
            return False
        vertex_set_match = self.vertices == other.vertices
        if not(vertex_set_match):
            return False
        edge_list_match = self.edges == other.edges
        if not(edge_list_match):
            return False
        vertex_names_match = all({self.vertex_names[i] == other.vertex_names[i] for i in self.vertices})
        if not(vertex_names_match):
            return False
        edge_names_match = all([self.edge_names[i] == other.edge_names[i] for i in range(0,self.next_edge_label)])
        return edge_names_match

class QuiverRepresentation:
    def __init__(self, quiver: Quiver, edge_matrices : Dict[str,np.array]):
        self._my_quiver = quiver
        self._edge_matrices = edge_matrices
        self.vertex_dims = {}
        edge_names_needed = set(quiver.edge_names.values())
        for k,v in edge_matrices.items():
            (rows,cols) = v.shape
            edge_idx = quiver.from_edge_name(k)
            if edge_idx is None:
                pass
            else:
                edge_names_needed.remove(k)
                (edge_src,edge_tgt) = quiver.edges[edge_idx]
                src_dim = self.vertex_dims.get(edge_src,None)
                if src_dim is None:
                    self.vertex_dims[edge_src] = cols
                elif src_dim==cols:
                    pass
                else:
                    raise ValueError("The edge matrices did not have a consistent dimension vector for all the vertices")
                tgt_dim = self.vertex_dims.get(edge_tgt,None)
                if tgt_dim is None:
                    self.vertex_dims[edge_tgt] = rows
                elif tgt_dim==rows:
                    pass
                else:
                    raise ValueError("The edge matrices did not have a consistent dimension vector for all the vertices")
        if len(edge_names_needed)>0:
            raise ValueError("Not all the edges had a matrix specified")

    def mat_from_path_alg(self,potential : 'PathAlgebra') -> np.array:
        assert self._my_quiver == potential._quiver
        return sum([np.multiply(self.to_matrix(k),v) for (k,v) in potential.element.element.items()])
    def to_matrix(self,word : str) -> np.array:
        factors = word.split(";")
        if len(factors)==0:
            raise ValueError("Only for paths of nonzero length")
        return reduce(lambda acc,edge_name: np.matmul(acc,self._edge_matrices[edge_name]),factors[1:],self._edge_matrices[factors[0]])

def FormalLinearCombination(t : type, base_ring : type, zero_base_ring : 'base_ring', ts_mul : Callable[['t','t'],'t']):
    # t is any type, if you want __mul__ to have meaning then you have to say it is a semigroup with the provided ts_mul for t's multiplication
    #   if you don't intend to call __mul__ ever you can put something dummy there like ts_mul = lambda z1,z2 = z2, but it will never be utilized
    # base_ring is some other type and it should have the __add__, __sub__ and __mul__ dunder methods work on it
    #   zero_base_ring also must be provided
    # the type annotations with 'base_ring' and 't' don't do anything. Only for documentation purposes.
    class FormalLinearCombinationT:
        def __init__(self, element_list : List[Tuple[base_ring,t]]):
            self.type = t
            self.base_ring = base_ring
            self.zero_base_ring = zero_base_ring
            self.ts_mul = ts_mul
            self.element = {e_i:coeff for (coeff,e_i) in element_list}
            self.my_class = type(self)
        def __iadd__(self,other):
            for x in self.element.keys():
                self.element[x] += other.element.get(x,self.zero_base_ring)
            for x,v in other.element.items():
                if self.element.get(x,None) is None:
                    self.element[x] = v
            return self
        def __add__(self,other):
            ret_class = self.my_class
            ret_val = ret_class([])
            for x_i,coeff_i in self.element.items():
                ret_val.element[x_i] = coeff_i
            for y_j,coeff_j in other.element.items():
                if ret_val.element.get(y_j,None) is None:
                    ret_val.element[y_j] = coeff_j
                else:
                    ret_val.element[y_j] += coeff_j
            return ret_val
        def __sub__(self,other):
            ret_class = self.my_class
            ret_val = ret_class([])
            for x_i,coeff_i in self.element.items():
                ret_val.element[x_i] = coeff_i
            for y_j,coeff_j in other.element.items():
                if ret_val.element.get(y_j,None) is None:
                    ret_val.element[y_j] = -coeff_j
                else:
                    ret_val.element[y_j] -= coeff_j
            return ret_val
        def __isub__(self,other):
            for x in self.element.keys():
                self.element[x] -= other.element.get(x,self.zero_base_ring)
            for x,v in other.element.items():
                if self.element.get(x,None) is None:
                    self.element[x] = -v
            return self
        def __mul__(self,other):
            if isinstance(other,self.base_ring):
                ret_class = self.my_class
                ret_val = ret_class([])
                for x_i,coeff_i in self.element.items():
                    ret_val.element[x_i] = coeff_i*other
                return ret_val
            else:
                ret_class = self.my_class
                ret_val = ret_class([])
                for x_i,coeff_i in self.element.items():
                    for y_j,coeff_j in other.element.items():
                        cur_key = self.ts_mul(x_i,y_j)
                        if ret_val.element.get(cur_key,None) is None:
                            ret_val.element[cur_key] = coeff_i*coeff_j
                        else:
                            ret_val.element[cur_key] += coeff_i*coeff_j
                return ret_val
        def remove_zero_terms(self,is_nonzero : Callable) -> None:
            self.element = {k:v for (k,v) in self.element.items() if is_nonzero(k) and not(v==self.zero_base_ring)}
        def __repr__(self) -> str:
            summands = [f"{v}*{k}" for (k,v) in self.element.items()]
            if len(summands)==0:
                return f"{self.zero_base_ring}"
            else:
                return "+".join(summands)
    FormalLinearCombinationT.__name__ = f"C[{t}]"
    return FormalLinearCombinationT

class PathAlgebra:
    def __init__(self, q: Quiver, path_combination : List[Tuple[complex,List[Nat]]]):
        self._quiver = q
        path_name = lambda path_list : ";".join([q.edge_names[i] for i in path_list])
        FLC_class = FormalLinearCombination(str, complex, complex(0,0), lambda z1,z2 : ";".join([z1,z2]) )
        self.element = FLC_class([(coeff,path_name(path_list)) for (coeff,path_list) in path_combination])
    def __add__(self,other:'PathAlgebra'):
        assert self._quiver==other._quiver
        new_element = self.element + other.element
        ret_val = PathAlgebra(self._quiver,[])
        ret_val.element = new_element
        return ret_val
    def __sub__(self,other:'PathAlgebra'):
        assert self._quiver==other._quiver
        new_element = self.element - other.element
        ret_val = PathAlgebra(self._quiver,[])
        ret_val.element = new_element
        return ret_val
    def __iadd__(self,other:'PathAlgebra'):
        assert self._quiver==other._quiver
        self.element += other.element
        return self
    def __isub__(self,other:'PathAlgebra'):
        assert self._quiver==other._quiver
        self.element -= other.element
        return self
    def is_nonzero(self,k : str) -> bool:
        factors = k.split(";")
        if len(factors)==0:
            raise ValueError("Only for paths of nonzero length")
        for (edge_before_name,edge_now_name) in zip(factors,factors[1:]):
            (edge_before_src,edge_before_tgt) = self._quiver.edges[self._quiver.from_edge_name(edge_before_name)]
            (edge_now_src,edge_now_tgt) = self._quiver.edges[self._quiver.from_edge_name(edge_now_name)]
            if not(edge_now_src==edge_before_tgt):
                return False
        return True
    def is_cyclic(self) -> bool:
        for k,v in self.element.element.items():
            factors = k.split(";")
            if len(factors)==0:
                raise ValueError("Each summand should have nonzero length")
            else:
                first_edge_name = factors[0]
                last_edge_name = factors[-1]
                (first_edge_src,first_edge_tgt) = self._quiver.edges[self._quiver.from_edge_name(first_edge_name)]
                (last_edge_src,last_edge_tgt) = self._quiver.edges[self._quiver.from_edge_name(last_edge_name)]
                if not(first_edge_src==last_edge_tgt) and not(v==self.base_ring_zero) and self.is_nonzero(k):
                    return False
        return True
    def split_cyclic(self) -> Tuple['PathAlgebra','PathAlgebra']:
        cyclic_part = self.element*complex(0,0)
        cyclic_part.remove_zero_terms(lambda k : self.is_nonzero(k))
        noncyclic_part = self.element*complex(0,0)
        noncyclic_part.remove_zero_terms(lambda k : self.is_nonzero(k))
        FLC_class = FormalLinearCombination(str, complex, complex(0,0), lambda z1,z2 : ";".join([z1,z2]) )
        for k,v in self.element.element.items():
            factors = k.split(";")
            if len(factors)==0:
                raise ValueError("Each summand should have nonzero length")
            else:
                first_edge_name = factors[0]
                last_edge_name = factors[-1]
                (first_edge_src,first_edge_tgt) = self._quiver.edges[self._quiver.from_edge_name(first_edge_name)]
                (last_edge_src,last_edge_tgt) = self._quiver.edges[self._quiver.from_edge_name(last_edge_name)]
                if first_edge_src==last_edge_tgt:
                    cyclic_part += FLC_class([(v,k)])
                else:
                    noncyclic_part += FLC_class([(v,k)])
        cyclic_part_wrapped = PathAlgebra(self._quiver,[])
        cyclic_part_wrapped.element = cyclic_part
        noncyclic_part_wrapped = PathAlgebra(self._quiver,[])
        noncyclic_part_wrapped.element = noncyclic_part
        return (cyclic_part_wrapped,noncyclic_part_wrapped)
    def cyclic_derivative(self, wrt_edge : Union[str,'FormalLinearCombinationT']) -> 'PathAlgebra':
        # cyclic derivative with respect to either the dual basis vector which is 1 only on the edge labelled=wrt_edge
        # or wrt_edge is a linear combination of such strings and the derivative extends linearly
        #   in a different language would be strict on the type of wrt_edge to be Either str (FormalLinearCombination<str>)
        if isinstance(wrt_edge,str):
            new_element = self.element*complex(0,0)
            new_element.remove_zero_terms(lambda k : self.is_nonzero(k))
            FLC_class = FormalLinearCombination(str, complex, complex(0,0), lambda z1,z2 : ";".join([z1,z2]) )
            for k,v in self.element.element.items():
                factors = k.split(";")
                if len(factors)==0:
                    raise ValueError("Each summand should have nonzero length")
                else:
                    for idx in range(0,len(factors)):
                        if factors[idx]==wrt_edge:
                            rotated = factors[(idx+1):] + factors[0:idx]
                            new_element += FLC_class([(v,";".join(rotated))])
            ret_val = PathAlgebra(self._quiver,[])
            ret_val.element = new_element
            return ret_val
        else:
            new_element = self.element*complex(0,0)
            new_element.remove_zero_terms(lambda k : self.is_nonzero(k))
            ret_val = PathAlgebra(self._quiver,[])
            ret_val.element = new_element
            for (k,v) in wrt_edge.element.items():
                ret_val += self.cyclic_derivative(k)*v
            return ret_val
    def __mul__(self,other:Union['PathAlgebra' , float , int , complex]):
        scalar_mult = False
        if isinstance(other,float):
            other_complex = complex(other,0)
            scalar_mult = True
        elif isinstance(other,int):
            other_complex = complex(other,0)
            scalar_mult = True
        elif isinstance(other,complex):
            other_complex = other
            scalar_mult = True
        if scalar_mult:
            new_element = self.element * other_complex
            ret_val = PathAlgebra(self._quiver,[])
            ret_val.element = new_element
            return ret_val
        else:
            assert self._quiver==other._quiver
            new_element = self.element * other.element
            new_element.remove_zero_terms(lambda k : self.is_nonzero(k))
            ret_val = PathAlgebra(self._quiver,[])
            ret_val.element = new_element
            return ret_val
    def __imul__(self,other:Union['PathAlgebra' , float , int , complex]):
        scalar_mult = False
        if isinstance(other,float):
            other_complex = complex(other,0)
            scalar_mult = True
        elif isinstance(other,int):
            other_complex = complex(other,0)
            scalar_mult = True
        elif isinstance(other,complex):
            other_complex = other
            scalar_mult = True
        if scalar_mult:
            self.element *= other_complex
        else:
            assert self._quiver==other._quiver
            self.element *= other.element
            self.element.remove_zero_terms(lambda k : self.is_nonzero(k))
        return self
    def __repr__(self):
        return f"{self.element} on {self._quiver}"

if __name__ == '__main__':
    jordan_quiver = Quiver()
    v_idx=jordan_quiver.add_vertex("alpha")
    a_idx=jordan_quiver.add_edge(v_idx,v_idx,"a")
    print(jordan_quiver)
    SumStr = FormalLinearCombination(str,complex,complex(0,0),lambda z1,z2 : z1+z2)
    flc_1a2b3c = SumStr([(complex(1,0),"a"),(complex(2,0),"b"),(complex(3,0),"c")])
    print(flc_1a2b3c)
    print(flc_1a2b3c+flc_1a2b3c)
    print(flc_1a2b3c*flc_1a2b3c)
    x = PathAlgebra(jordan_quiver,[(complex(1,0),[a_idx])])
    print(x)
    print(x+x)
    print(x*x)
    kronecker_quiver = Quiver()
    v1=kronecker_quiver.add_vertex("alpha")
    v2=kronecker_quiver.add_vertex("beta")
    a_idx=kronecker_quiver.add_edge(v1,v2,"a")
    b_idx=kronecker_quiver.add_edge(v1,v2,"b")
    xa = PathAlgebra(kronecker_quiver,[(complex(1,0),[a_idx])])
    xb = PathAlgebra(kronecker_quiver,[(complex(1,0),[b_idx])])
    print(xa-xb*5)
    print(xa*xb)
    y = xa
    print(y)
    y*=complex(0,1)
    print(y)
    y*=xb
    print(y)
    triple_quiver = Quiver()
    v_idx=triple_quiver.add_vertex("alpha")
    a_idx=triple_quiver.add_edge(v_idx,v_idx,"a")
    b_idx=triple_quiver.add_edge(v_idx,v_idx,"adag")
    c_idx=triple_quiver.add_edge(v_idx,v_idx,"omega")
    xa = PathAlgebra(triple_quiver,[(complex(1,0),[a_idx])])
    xb = PathAlgebra(triple_quiver,[(complex(1,0),[b_idx])])
    xc = PathAlgebra(triple_quiver,[(complex(1,0),[c_idx])])
    ginz_cubic = xa*xb*xc-xb*xa*xc
    print(f"Ginzburg cubic [a,a^dagger]omega is cyclic? : {ginz_cubic.is_cyclic()}")
    ginz_cyclic_deriv = ginz_cubic.cyclic_derivative("omega")
    print(f"Ginzburg cubic [a,a^dagger]omega cyclic derivative wrt omega : {ginz_cyclic_deriv}")
    ginz_cyclic_deriv_2 = ginz_cubic.cyclic_derivative(SumStr([(complex(5,0),"omega"),(complex(2,0),"a")]))
    print(f"Ginzburg cubic [a,a^dagger]omega cyclic derivative wrt 5*omega+2*a : {ginz_cyclic_deriv_2}")
    (cyc,noncyc) = ginz_cyclic_deriv_2.split_cyclic()
    print(f"It's cyclic part is {cyc}.\nIt's noncyclic part is {noncyc}")
    edge_dict = {}
    edge_dict["a"] = np.array([[0,0],[1,0]])
    edge_dict["adag"] = np.array([[0,1],[0,0]])
    edge_dict["omega"] = np.array([[1,0],[0,1]])
    qrep = QuiverRepresentation(triple_quiver,edge_dict)
    print(f"The dimension vector for the constructed representation on {triple_quiver} : {qrep.vertex_dims}")
    ginz_cubic_mat = qrep.mat_from_path_alg(ginz_cubic)
    print(ginz_cubic_mat)