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

def test_perm_promoted() -> None:
    """
    S_10 included into vJ_10
    """
    MY_N = 10
    perm_1 = Permutation.random(MY_N)
    perm_2 = Permutation.random(MY_N)
    perm_1_promoted = VirtualCactusGroup(MY_N)
    perm_1_promoted *= perm_1
    perm_2_promoted = VirtualCactusGroup(MY_N)
    perm_2_promoted *= perm_2
    perm_12_promoted = VirtualCactusGroup(MY_N)
    perm_12_promoted *= (perm_1*perm_2)
    assert perm_1_promoted*perm_2_promoted == perm_12_promoted

#pylint:disable=too-many-branches,too-many-statements
def test_cross_rels() -> None:
    """
    the defining ws_ijw^-1 = s_wi,wj relations in vJ_5
    """
    MY_N = 5

    s_gens: Dict[Tuple[int, int], VirtualCactusGroup] = {}
    iden = VirtualCactusGroup(MY_N)
    # define the s_ij
    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            current_gen = CactusGroup(MY_N)
            current_gen.append_generator(p, q)
            current_vgen = iden*current_gen
            s_gens[(p, q)] = current_vgen

    perm_1 = Permutation.random(MY_N)
    perm_1_promoted = VirtualCactusGroup(MY_N)
    perm_1_promoted *= perm_1

    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            maybe_k_shift = perm_1.preserves_intervalness(p,q)
            if maybe_k_shift is None:
                continue
            expected_rhs = VirtualCactusGroup(MY_N)
            expected_rhs_cactus = s_gens[(p+maybe_k_shift,q+maybe_k_shift)]
            expected_rhs *= expected_rhs_cactus
            spq = s_gens[(p, q)]
            lhs = perm_1_promoted*spq
            lhs /= perm_1_promoted
            assert lhs==expected_rhs

    perm_1.element = list(range(1,MY_N+1))
    perm_1.element[0], perm_1.element[1] = perm_1.element[1], perm_1.element[0]
    perm_1_promoted = VirtualCactusGroup(MY_N)
    perm_1_promoted *= perm_1

    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            maybe_k_shift = perm_1.preserves_intervalness(p,q)
            #pylint:disable=no-else-continue
            if maybe_k_shift is None:
                #pylint:disable=consider-using-in
                assert p==1 or p==2
                continue
            else:
                assert p>2
                assert maybe_k_shift == 0
            expected_rhs = VirtualCactusGroup(MY_N)
            expected_rhs_cactus = s_gens[(p+maybe_k_shift,q+maybe_k_shift)]
            expected_rhs *= expected_rhs_cactus
            spq = s_gens[(p, q)]
            lhs = perm_1_promoted*spq
            lhs /= perm_1_promoted
            assert lhs==expected_rhs

    perm_1.element = list(range(3,MY_N+3))
    perm_1.element[-1] = 2
    perm_1.element[-2] = 1
    perm_1_promoted = VirtualCactusGroup(MY_N)
    perm_1_promoted *= perm_1

    for p in range(1, MY_N):
        for q in range(p+1, MY_N+1):
            maybe_k_shift = perm_1.preserves_intervalness(p,q)
            #pylint:disable=no-else-continue
            if maybe_k_shift is None:
                continue
            else:
                #pylint:disable=consider-using-in
                assert maybe_k_shift == 2 or maybe_k_shift == -3
            expected_rhs = VirtualCactusGroup(MY_N)
            expected_rhs_cactus = s_gens[(p+maybe_k_shift,q+maybe_k_shift)]
            expected_rhs *= expected_rhs_cactus
            spq = s_gens[(p, q)]
            lhs = perm_1_promoted*spq
            lhs /= perm_1_promoted
            assert lhs==expected_rhs

def test_random_iden() -> None:
    """
    produce a random element of vJ_4 and check that
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
            # it might not simplify all the way to identity
            print(f"{lhs_vcact} with {my_len}")
