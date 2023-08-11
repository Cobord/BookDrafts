"""
test triangulations of convex m-gons
"""
#pylint:disable=import-error,invalid-name,R0801
from math import pi as PI,sin,cos
from triangulation import Triangulation

def test_octagon() -> None:
    """
    a triangulation of an octagon
    """
    N = 8
    my_diagonals = [(4,7),(2,8),(2,4),(2,7),(4,6)]
    RADIUS = 2
    t = Triangulation([ (RADIUS*cos(2*PI*which/N),RADIUS*sin(2*PI*which/N))
                       for which in range(N)], [(x-1,y-1) for x,y in my_diagonals])
    diags_before = t.my_diagonal_list.copy()
    for diag,quad in zip(t.my_diagonal_list,t.my_quadrilaterals):
        assert diag[0]==quad[0]
        assert diag[1]==quad[2]
    p = t.to_plabic()
    assert p.my_perfect_matching is not None
    change, new_diag = t.quad_flip((1,3))
    assert change
    assert new_diag == (2,6)
    assert change
    assert new_diag == (2,6)
    for which_diag_now,diag in enumerate(t.my_diagonal_list):
        if diag != (2,6):
            assert diag == diags_before[which_diag_now]
        else:
            assert diags_before[which_diag_now] == (1,3)
    for diag,quad in zip(t.my_diagonal_list,t.my_quadrilaterals):
        assert diag[0]==quad[0]
        assert diag[1]==quad[2]
    p = t.to_plabic()
    assert p.my_perfect_matching is not None
