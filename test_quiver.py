"""
test quivers
"""
#pylint:disable=import-error,line-too-long,invalid-name
from sympy import symbols
from quiver import Quiver, FormalLinearCombination,PathAlgebra

def test_linear_comb() -> None:
    """
    formal linear combination of strings with complex coefficients
    """
    SumStr = FormalLinearCombination(
        str, complex, complex(0, 0), lambda z1, z2: z1+z2)
    flc_1a2b3c = SumStr(
        [(complex(1, 0), "a"), (complex(2, 0), "b"), (complex(3, 0), "c")])
    assert str(flc_1a2b3c) == "(1+0j)*a+(2+0j)*b+(3+0j)*c"
    two_flc_1a2b3c = flc_1a2b3c+flc_1a2b3c
    exp_two_flc_1a2b3c = SumStr(
        [(complex(2, 0), "a"), (complex(4, 0), "b"), (complex(6, 0), "c")])
    assert str(two_flc_1a2b3c) == "(2+0j)*a+(4+0j)*b+(6+0j)*c"
    assert exp_two_flc_1a2b3c == two_flc_1a2b3c
    assert str(flc_1a2b3c*flc_1a2b3c) == "(1+0j)*aa+(2+0j)*ab+(3+0j)*ac"+\
        "+(2+0j)*ba+(4+0j)*bb+(6+0j)*bc"+\
        "+(3+0j)*ca+(6+0j)*cb+(9+0j)*cc"

def test_jordan() -> None:
    """
    test construction and string of Jordan quiver
    and of elements of it's path algebra
    """
    jordan_quiver = Quiver()
    v_name = "alpha"
    v_idx = jordan_quiver.add_vertex(v_name)
    e_name = "a"
    a_idx = jordan_quiver.add_edge(v_idx, v_idx, e_name)
    assert str(jordan_quiver)==\
        f"Quiver with vertices {{\'{v_name}\'}} and edges [\'Edge {e_name} : {v_name} -> {v_name}\']"
    coeff = complex(1,0)
    x = PathAlgebra(jordan_quiver, [(coeff, [a_idx])])
    assert str(x) == f"{coeff}*{e_name} on {jordan_quiver}"
    a_sym = symbols("a",commutative=False)
    assert x.to_symbolic() == coeff*a_sym
    y = x+x
    assert str(y) == f"{2*coeff}*{e_name} on {jordan_quiver}"
    y = x*x
    assert str(y) == f"{coeff**2}*{e_name};{e_name} on {jordan_quiver}"
