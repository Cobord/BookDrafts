"""
test for cactus group
"""
#pylint:disable=invalid-name,too-many-locals,import-error
from typing import Tuple, Dict
from cactus import CactusGroup, Permutation, VirtualCactusGroup, perm_multiply

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

def test_vCactus() -> None:
    """
    check the generators and relations of J_10 included into vJ_10
    """
    MY_N = 10
    s_gens: Dict[Tuple[int, int], VirtualCactusGroup] = {}
    iden = VirtualCactusGroup(MY_N)
    # define the generators
    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            current_gen = CactusGroup(MY_N)
            current_gen.append_generator(p, q)
            current_vgen = iden*current_gen
            s_gens[(p, q)] = current_vgen

    # check the relations and
    # check that to_permutation
    # obeys the defining relations
    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            spq = s_gens[(p, q)]
            lhs = spq*spq
            assert lhs == iden, f"{str(spq)} and {str(iden)}"
            for k in range(q+1, MY_N):
                for l in range(k+1, MY_N+1):
                    skl = s_gens[(k, l)]
                    lhs = spq*skl
                    rhs = skl*spq
                    str1 = str(lhs)
                    str2 = str(rhs)
                    assert lhs == rhs, f"{str1} and {str2}"
            for k in range(p, MY_N):
                for l in range(k+1, q+1):
                    skl = s_gens[(k, l)]
                    s_messy = s_gens[(p+q-l, p+q-k)]
                    assert spq*skl == s_messy*spq

def test_random_iden() -> None:
    """
    produce a random element of vJ_3 and check that
    x*x^-1 gives identity
    """
    MY_N = 4
    iden_perm = Permutation(MY_N)
    iden_cact = CactusGroup(MY_N)
    iden_v = VirtualCactusGroup(MY_N)

    for my_len in range(7):
        lhs = CactusGroup.random(MY_N,my_len)
        lhs *= lhs.inv()
        assert lhs.to_permutation() == iden_perm.element, f"{lhs}"
        if lhs != iden_cact:
            lhs.simplify()
            assert lhs == iden_cact, f"{lhs}"

    for _ in range(10):
        lhs_perm = Permutation.random(MY_N)
        lhs_perm *= lhs_perm.inv()
        assert lhs_perm == iden_perm, f"{lhs_perm}"

    MY_CACT_LEN = 6
    for my_len in [0,1,2]:
        lhs_vcact = VirtualCactusGroup.random(MY_N,MY_CACT_LEN,my_len)
        lhs_vcact *= lhs_vcact.inv()
        if lhs_vcact != iden_v:
            print(f"{lhs_vcact} with {my_len}")
