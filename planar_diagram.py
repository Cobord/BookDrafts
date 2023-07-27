"""
diagrams used in the sense of cluster algebra
"""

from typing import Iterator, List, Set, Tuple, TypeVar, Union, runtime_checkable, Protocol
import itertools
from functools import reduce
import networkx as nx

NT = TypeVar("NT")
WeightType = Union[float, NT]
EdgeData = Tuple[int, int, WeightType]
Nat = int

# pylint:disable=line-too-long,invalid-name,too-many-locals,no-member,too-many-branches,too-few-public-methods


class PlanarNetwork(nx.DiGraph):
    """
    built on top of a directed graph
    """

    def __init__(self, n: Nat, edge_list: List[List[EdgeData]], *, multiplicative_identity: WeightType = 1.0, additive_identity: WeightType = 0.0):
        """
        a planar diagram with n horizontal lines
        """
        super().__init__()
        self.multiplicative_identity = multiplicative_identity
        self.additive_identity = additive_identity
        for x in range(1, n+1):
            self.add_node(f"{x}_0")
        sink_number = {x: 0 for x in range(1, n+1)}
        for stage in edge_list:
            already_incremented = []
            for edge in stage:
                from_node, to_node, cur_weight = edge
                if from_node in already_incremented:
                    from_number = sink_number[from_node]-1
                else:
                    from_number = sink_number[from_node]
                if to_node in already_incremented:
                    to_number = sink_number[to_node]
                else:
                    to_number = sink_number[to_node]+1
                    sink_number[to_node] += 1
                    self.add_edge(f"{to_node}_{to_number-1}",
                                  f"{to_node}_{to_number}", weight=multiplicative_identity)
                    already_incremented.append(to_node)
                self.add_node(f"{to_node}_{to_number}")
                self.add_edge(f"{from_node}_{from_number}",
                              f"{to_node}_{to_number}", weight=cur_weight)
        self.sink_number = sink_number

    def path_weight(self, path_nodes) -> WeightType:
        """
        multiply the weights for the path that connects all the path_nodes
        """
        path_edges = [self[path_nodes[i]][path_nodes[i+1]]
                      for i in range(len(path_nodes)-1)]
        return reduce(lambda x, y: x*y, [x['weight'] for x in path_edges], self.multiplicative_identity)

    def weight_matrix(self, my_i: Nat, my_j: Nat) -> WeightType:
        """
        the sum of the path weights over all simple paths connecting
        the source node on horizontal line i
        the sink node on horizontal line j
        """
        source_node = f"{my_i}_0"
        sink_node = f"{my_j}_{self.sink_number[my_j]}"
        weight_ij = self.additive_identity
        for path_nodes in nx.all_simple_paths(self, source_node, sink_node):
            cur_contrib = self.path_weight(path_nodes)
            weight_ij += cur_contrib
        return weight_ij

    def vertex_disjoint_collection(self, i_set: Set[Nat], j_set: Set[Nat]) -> Iterator[WeightType]:
        """
        iterate through all systems of paths
        that connect the sources with i_set to the sinks with j_set
        suppose there are k of each so we get k paths each time
        skip if they are not vertex disjoint
        but if they are vertex disjoint give the product of the path weights
        """
        # start off by finding the possibilities of how i_set and j_set match up
        # so that (i,j) in collection means that particular i in i_set connects to j in j_set
        if len(i_set) == 0 or len(j_set) == 0:
            raise ValueError("The sets of sources and sinks must be nonempty")
        relevant_pairings = [pairings_nc(i_set,j_set)]
        # relevant_pairings = pairings(i_set,j_set)
        for collection in relevant_pairings:
            path_collection_iterator = None
            # build up path_collection_iterator by taking the product of iterators
            # where each factor iterator is the ones that connect i to j
            ij_paths = (nx.all_simple_paths(
                    self, f"{cur_i}_0", f"{cur_j}_{self.sink_number[cur_j]}")
                    for cur_i, cur_j in collection)
            path_collection_iterator = itertools.product(*ij_paths)
            # now path_collection is a tuple of k paths that fully connect i_set to j_set
            for path_collection in path_collection_iterator:
                seen_vertices = set({})
                is_vertex_disjoint = True
                my_paths = []
                for path in path_collection:
                    my_paths.append(list(path))
                    seen_vertices.isdisjoint(set(path))
                    if seen_vertices.isdisjoint(set(path)):
                        seen_vertices = seen_vertices.union(set(path))
                    else:
                        is_vertex_disjoint = False
                        break
                if is_vertex_disjoint:
                    all_path_weights = [self.path_weight(
                        path_nodes) for path_nodes in my_paths]
                    yield reduce(lambda x, y: x*y, all_path_weights, self.multiplicative_identity)

    def lindstrom_minor(self, i_set: Set[Nat], j_set: Set[Nat]) -> WeightType:
        """
        the minor where specify to include the rows and columns specified by i_set and j_set
        """
        minor_value = self.additive_identity
        for contrib in self.vertex_disjoint_collection(i_set, j_set):
            minor_value += contrib
        return minor_value


T = TypeVar("T")
U = TypeVar("U")


def pairings(i_set: Set[T], j_set: Set[U]) -> Iterator[Set[Tuple[T, U]]]:
    """
    generator that goes through all pairings of i_set with j_set
    """
    if len(i_set) != len(j_set):
        raise ValueError("Must have same number of elements")
    if len(i_set) == 0:
        yield set({})
        return
    xtemp = i_set.copy()
    i_popped = xtemp.pop()
    for my_i, my_j in {(i_popped, j) for j in j_set}:
        ytemp = j_set.copy()
        ytemp.remove(my_j)
        for remaining in pairings(xtemp, ytemp):
            remaining.add((my_i, my_j))
            yield remaining


@runtime_checkable
class Ordered(Protocol):
    """
    protocol for ordered
    """

    def __leq__(self, other): ...


V = TypeVar('V', bound=Ordered)
W = TypeVar('W', bound=Ordered)

def pairings_nc(i_set: Set[V], j_set: Set[W]) -> Set[Tuple[V, W]]:
    """
    the unique pairing of i_set with j_set
    such that the minimum of i_set is matched with the minimum of j_set
    and so on all the way up
    """
    if len(i_set) != len(j_set):
        raise ValueError("Must have same number of elements")
    i_list = list(i_set)
    i_list.sort()
    j_list = list(j_set)
    j_list.sort()
    return set(zip(i_list, j_list))

# pylint:disable = too-many-arguments
def determinant(matrix, mul, simplifier, additive_identity,negative_one,simplify_freq=3,original_width=None):
    """
    determinant using only + and *
    so that it can work on complex object types
    that override __add__ and __mul__
    imported det functions assume primitive numeric type like float
    """
    width = len(matrix)
    if original_width is None:
        original_width = width
    if width == 1:
        to_return = mul * matrix[0][0]
        if original_width == width:
            to_return = simplifier(to_return)
        return to_return
    sign = negative_one
    answer = additive_identity
    for skip_col in range(width):
        m = []
        for jdx in range(1, width):
            buff = []
            for kdx in range(width):
                if kdx != skip_col:
                    buff.append(matrix[jdx][kdx])
            m.append(buff)
        sign *= negative_one
        answer = answer + mul * determinant(m, sign * matrix[0][skip_col],simplifier,additive_identity,negative_one,simplify_freq,original_width)
    if width % simplify_freq == original_width % simplify_freq:
        answer = simplifier(answer)
    return answer

if __name__ == "__main__":
    # example is taken from https://arxiv.org/pdf/math/9912128.pdf figure 1
    from sympy import symbols, Symbol, Expr

    def annotated_symbols(*args, **kwargs) -> Tuple[Symbol, ...]:
        """
        because symbols is typed Any
        despite the fact that for the arguments we are using it returns an tuple of Symbols
        """
        return symbols(*args, **kwargs)
    A, B, C, D, E, F, G, H, I = annotated_symbols(
        'a,b,c,d,e,f,g,h,i', commutative=True)
    ZERO, ONE = annotated_symbols('zero,one', commutative=True)
    p = PlanarNetwork(3, [[(3, 2, A), (3, 3, ONE)], [(3, 2, C), (2, 1, B)], [
                      (2, 2, E), (1, 1, D)], [(3, 3, F), (2, 3, G), (1, 2, H)], [(2, 3, I)]],
                      multiplicative_identity=ONE,
                      additive_identity=ZERO)
    expected_weight_matrix: List[List[Expr]] = [
        [D, D*H, D*H*I], [B*D, B*D*H+E, B*D*H*I+E*(G+I)], [A*B*D, A*B*D*H+(A+C)*E, A*B*D*H*I+(A+C)*E*(G+I)+F]]
    for i in range(1, 4):
        for j in range(1, 4):
            w_ij: Expr = p.weight_matrix(i, j)
            w_ij = w_ij.subs({ZERO: 0.0, ONE: 1.0})
            print(f"a_({i},{j}) = {w_ij}")
            assert w_ij.equals(expected_weight_matrix[i-1][j-1])
    print("Weights of vertex disjoint collections")
    for weight in p.vertex_disjoint_collection({2, 3}, {2, 3}):
        print(f"\t{weight}")
    print("Doing Lindstrom")
    delta_23_23: Expr = p.lindstrom_minor({2, 3}, {2, 3})
    delta_23_23 = delta_23_23.subs({ZERO: 0.0, ONE: 1.0})
    print(f"Delta_23,23 = {delta_23_23}")
    expected = (B*C*D*E*G*H + B*D*F*H + E*F)*1.0
    assert delta_23_23 == expected
    slice_23 = (2,3)
    my_minor = [[expected_weight_matrix[cur_i-1][cur_j-1] for cur_i in slice_23] for cur_j in slice_23]
    expected_2 = determinant(my_minor,ONE,
                            lambda z : z.subs({ZERO: 0.0, ONE: 1.0}).simplify(),
                            ZERO,-ONE)
    assert expected_2 == expected

    delta_123_123: Expr = p.lindstrom_minor({2, 3, 1}, {1, 2, 3})
    delta_123_123 = delta_123_123.subs({ZERO: 0.0, ONE: 1.0})
    print(f"Delta_123,123 = {delta_123_123}")
    expected_determinant = determinant(expected_weight_matrix,
                                       ONE,
                                       lambda z : z.subs({ZERO: 0.0, ONE: 1.0}).simplify(),
                                       ZERO,
                                       -ONE)
    assert delta_123_123 == expected_determinant
