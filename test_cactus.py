"""
test for cactus group
"""
#pylint:disable=invalid-name,too-many-locals,import-error
from typing import Tuple, Dict
from cactus import CactusGroup, perm_multiply

def test_gen_rel_cactus() -> None:
    """
    check the generators and relations of J_10
    """
    MY_N = 10
    s_gens: Dict[Tuple[int, int], CactusGroup] = {}
    iden = CactusGroup(MY_N)
    # define the generators
    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            current_gen = CactusGroup(MY_N)
            current_gen.append_generator(p, q)
            s_gens[(p, q)] = current_gen

    # check the relations and
    # check that to_permutation
    # obeys the defining relations
    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            spq = s_gens[(p, q)]
            assert spq*spq == iden
            pi_pq = spq.to_permutation()
            for idx in range(0, p-1):
                assert pi_pq[idx] == idx+1, f"{p} {q} {pi_pq} {idx}"
            for idx in range(q, spq.index):
                assert pi_pq[idx] == idx+1, f"{p} {q} {pi_pq} {idx}"
            for idx in range(p-1, q):
                assert pi_pq[idx] == q-(idx-p+1), f"{p} {q} {pi_pq} {idx}"
            assert perm_multiply(pi_pq, pi_pq) == iden.to_permutation()
            for k in range(q+1, MY_N):
                for l in range(k+1, MY_N+1):
                    skl = s_gens[(k, l)]
                    pi_kl = skl.to_permutation()
                    assert spq*skl == skl*spq
                    assert perm_multiply(
                        pi_pq, pi_kl) == perm_multiply(pi_kl, pi_pq)
            for k in range(p, MY_N):
                for l in range(k+1, q+1):
                    skl = s_gens[(k, l)]
                    pi_kl = skl.to_permutation()
                    s_messy = s_gens[(p+q-l, p+q-k)]
                    pi_messy = s_messy.to_permutation()
                    assert spq*skl == s_messy*spq
                    lhs = perm_multiply(pi_pq, pi_kl)
                    rhs = perm_multiply(pi_messy, pi_pq)
                    assert lhs == rhs
