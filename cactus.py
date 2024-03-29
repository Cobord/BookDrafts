"""
cactus group J_n
"""

from __future__ import annotations
from functools import reduce
from random import shuffle, sample
from typing import Optional, List, Tuple, Union, cast

# pylint:disable = line-too-long


class CactusGroup:
    """
    elements of J_n for some n
    """

    def __init__(self, my_n: int):
        """
        initialize as the identity element of J_n
        """
        self.name = f"J_{my_n}"
        self.index = my_n
        self.element: List[Tuple[int, int]] = []

    def __str__(self) -> str:
        return str(self.element)

    def to_permutation(self) -> List[int]:
        """
        the image under the J_n to S_n morphism
        """
        def to_permutation_helper(my_pq: Tuple[int, int]) -> List[int]:
            """
            the image of the generators
            """
            my_p, my_q = my_pq
            my_p -= 1
            my_q -= 1
            return_permutation = list(range(0, self.index))
            halfway = (my_q-my_p)//2+1
            for help_idx in range(0, halfway):
                if my_p+help_idx >= my_q-help_idx:
                    break
                return_permutation[my_p+help_idx], return_permutation[my_q -
                                                                      help_idx] = return_permutation[my_q-help_idx], return_permutation[my_p+help_idx]
            return return_permutation

        def perm_multiply_0(perm_1: List[int], perm_2: List[int]) -> List[int]:
            """
            multiply permutations 0 through n-1
            """
            return [perm_1[perm_2_i] for perm_2_i in perm_2]
        zero_indexed_answer = reduce(perm_multiply_0, (to_permutation_helper(
            z) for z in self.element), list(range(0, self.index)))
        return list((z+1 for z in zero_indexed_answer))

    def append_generator(self, my_p: int, my_q: int):
        """
        multiply the current element by s_pq
        """
        if (my_p < 1 or my_p > self.index) or (my_q < 1 or my_q > self.index) or (my_q <= my_p):
            raise ValueError(" ".join([f"p,q of s_p,q must be in the range from 1 to {self.index}",
                             "and p must be less than q"]))
        self.element.append((my_p, my_q))
        self.simplify(only_last=True)

    def __imul__(self, other: CactusGroup):
        """
        group multiplication
        """
        if not self.name == other.name:
            raise TypeError("Elements of different groups")
        if len(self.element) == 0:
            self.element = other.element.copy()
        elif len(other.element) == 0:
            pass
        else:
            last_of_self = len(self.element)-1
            self.element.extend(other.element.copy())
            if last_of_self+1 < len(self.element):
                _ = self.simplify(
                    only_last=False, possible_pairs=[last_of_self])
        return self

    def __mul__(self, other: CactusGroup):
        ret_val = CactusGroup(self.index)
        ret_val *= self
        ret_val *= other
        return ret_val

    def copy(self) -> CactusGroup:
        """
        a copy
        """
        ret_val = CactusGroup(self.index)
        ret_val.element = self.element.copy()
        return ret_val

    def __eq__(self, other) -> bool:
        """
        is self*other^-1 equal to the identity
        can say they are not equal when they are depending on simplification
        """
        if not isinstance(other, CactusGroup):
            return False
        if not self.name == other.name:
            raise TypeError("Elements of different groups")
        self.simplify()
        other.simplify()
        self_is_iden = len(self.element) == 0
        other_is_iden = len(other.element) == 0
        if self_is_iden:
            other.simplify(only_last=False)
            return len(other.element) == 0
        if other_is_iden:
            self.simplify(only_last=False)
            return len(self.element) == 0
        other_inv = CactusGroup(other.index)
        other_inv.element = other.element
        other_inv.element.reverse()
        prod = self*other_inv
        _ = prod.simplify()
        return len(prod.element) == 0

    # pylint:disable = too-many-branches
    def simplify(self, only_last=False, possible_pairs: Optional[List[int]] = None) -> bool:
        """
        simplify the group element which is presented as a word in the s_p,q letters
        if only_last then the only possible location for a simplification to occur is the last pair
        the optional argument of possible_pairs which specifies
        where simplifications can occur, if not only_last then this defaults to
        all the neighboring pairs
        returns whether any simplification happened to self
        """
        if only_last and len(self.element) > 1:
            possible_pairs = [len(self.element)-2]
        elif possible_pairs is None:
            possible_pairs = list(range(len(self.element)-1))
        changed = False
        for try_here in possible_pairs:
            try:
                p_1, q_1 = self.element[try_here]
                p_2, q_2 = self.element[try_here+1]
            except IndexError:
                continue
            if p_1 == p_2 and q_1 == q_2:
                # an inverse pair next to each other cancels out
                changed = True
                self.element = self.element[0:try_here] + \
                    self.element[try_here+2:]
                possible_pairs = [(x if x < try_here else x-2)
                                  for x in possible_pairs if x < try_here or x > try_here+1]
                if try_here-1 >= 0 and try_here < len(self.element):
                    possible_pairs.append(try_here-1)
            elif q_2 < p_1:
                # a commuting pair, but in the wrong order for the normal form
                changed = True
                self.element[try_here], self.element[try_here +
                                                     1] = self.element[try_here+1], self.element[try_here]
                if try_here-1 >= 0 and try_here < len(self.element):
                    possible_pairs.append(try_here-1)
                if try_here+2 < len(self.element):
                    possible_pairs.append(try_here+1)
            elif p_1 <= p_2 and q_2 <= q_1:
                # the nested intervals case
                # use the most interesting defining relation of J_n
                changed = True
                self.element[try_here] = (p_1+q_1-q_2, p_1+q_1-p_2)
                self.element[try_here+1] = (p_1, q_1)
                if try_here-1 >= 0 and try_here < len(self.element):
                    possible_pairs.append(try_here-1)
                if try_here+2 < len(self.element):
                    possible_pairs.append(try_here+1)
        return changed

    @staticmethod
    def random(my_n : int,my_cact_len : int) -> CactusGroup:
        """
        produce a random element of J_{my_n}
        my_cact_len is a bound for how many generators to use
        """
        to_ret = CactusGroup(my_n)
        one_to_n = list(range(1,my_n+1,1))
        for _ in range(my_cact_len):
            tup = cast(Tuple[int,int],tuple(sorted(sample(one_to_n,k=2))))
            to_ret.append_generator(tup[0],tup[1])
        to_ret.simplify()
        return to_ret

    def inv_inplace(self) -> None:
        """
        replace self with it's inverse
        """
        self.element.reverse()

    def inv(self) -> CactusGroup:
        """
        return it's inverse
        """
        ret_val = self.copy()
        ret_val.inv_inplace()
        return ret_val

    def __itruediv__(self,other : CactusGroup) -> CactusGroup:
        self *= other.inv()
        return self

    def __div__(self,other : CactusGroup) -> CactusGroup:
        ret_val = CactusGroup(self.index)
        ret_val *= self
        ret_val /= other
        return ret_val

def perm_multiply(perm_1: List[int], perm_2: List[int]) -> List[int]:
    """
    multiply permutations 1 through n
    """
    return [perm_1[perm_2_i-1] for perm_2_i in perm_2]

class Permutation:
    """
    a permutaton
    """
    def __init__(self, my_n: int):
        self.name = f"S_{my_n}"
        self.index = my_n
        self.element: List[int] = list(range(1,my_n+1))

    def __str__(self) -> str:
        return str(self.element)

    def __imul__(self, other: Permutation):
        """
        group multiplication
        """
        if not self.name == other.name:
            raise TypeError("Elements of different groups")
        if len(self.element) == 0:
            self.element = other.element.copy()
        elif len(other.element) == 0:
            pass
        else:
            self.element = perm_multiply(self.element, other.element)
        return self

    def __mul__(self, other: Permutation):
        ret_val = Permutation(self.index)
        ret_val *= self
        ret_val *= other
        return ret_val

    def preserves_intervalness(self,i : int,j : int) -> Optional[int]:
        """
        self takes the closed interval i through j
        to another closed interval (i+k) through (j+k)
        preserving the ordering within this interval
        if so, return that k
        otherwise None
        """
        if i<1 or i>self.index or j<1 or j>self.index or j<i:
            raise ValueError(f"{i}-{j} not a valid interval for {self.name}")
        presumed_shift = self.element[i-1]-i
        for idx in range(i+1,j+1):
            cur_shift = self.element[idx-1]-idx
            if cur_shift != presumed_shift:
                return None
        return presumed_shift

    def copy(self) -> Permutation:
        """
        a copy
        """
        ret_val = Permutation(self.index)
        ret_val.element = self.element.copy()
        return ret_val

    def __eq__(self, other) -> bool:
        """
        is self*other^-1 equal to the identity
        can say they are not equal when they are depending on simplification
        """
        if not isinstance(other, Permutation):
            return False
        if not self.name == other.name:
            raise TypeError("Elements of different groups")
        return self.element == other.element

    @staticmethod
    def random(my_n : int) -> Permutation:
        """
        produce a random element of S_{my_n}
        """
        to_ret = Permutation(my_n)
        shuffle(to_ret.element)
        return to_ret

    def inv(self) -> Permutation:
        """
        return it's inverse
        """
        ret_val = self.copy()
        for idx in range(1,self.index+1):
            idx_comes_from = self.element.index(idx)+1
            ret_val.element[idx-1] = idx_comes_from
        return ret_val

    def __itruediv__(self,other : Permutation) -> Permutation:
        self *= other.inv()
        return self

    def __div__(self,other : Permutation) -> Permutation:
        ret_val = Permutation(self.index)
        ret_val *= self
        ret_val /= other
        return ret_val

class VirtualCactusGroup:
    """
    generated by s_{ij} of cactus
    and w of S_n
    and if w turns the interval ij to w(i)w(j)
    without any other effect on that interval
    can do whatever else outside

    implemented as alternating product
    of part from cactus group and part from symmetric group
    if one of them is the identity
    use multiplication in either to collapse 2 tuples into 1 tuple
    don't necessarily reduce along the ws_ijw-1 = s_wiwj relation
    """
    def __init__(self, my_n: int):
        self.name = f"vJ_{my_n}"
        self.index = my_n
        self.element: List[Tuple[CactusGroup, Permutation]] = []

    def __imul__(self, other: Union[VirtualCactusGroup , Permutation , CactusGroup]):
        """
        group multiplication
        """
        if isinstance(other,Permutation):
            if not self.index == other.index:
                raise TypeError("Elements of different groups")
            if len(self.element)>0:
                cact_part,perm_part = self.element[-1]
                perm_part*=other
                self.element[-1] = (cact_part,perm_part)
            else:
                self.element.append((CactusGroup(self.index),other))
        elif isinstance(other,CactusGroup):
            other_promoted = VirtualCactusGroup(other.index)
            other_promoted.element.append((other.copy(),Permutation(other.index)))
            self *= other_promoted
        else:
            if not self.name == other.name:
                raise TypeError("Elements of different groups")
            if len(self.element) == 0:
                other_copy = other.copy()
                self.element = other_copy.element
            elif len(other.element) == 0:
                pass
            else:
                last_of_self = len(self.element)-1
                other_copy = other.copy()
                self.element.extend(other_copy.element)
                if last_of_self+1 < len(self.element):
                    _ = self.simplify(
                        only_last=False, possible_pairs=[last_of_self])
        return self

    def __mul__(self, other: Union[VirtualCactusGroup , Permutation , CactusGroup]) -> VirtualCactusGroup:
        ret_val = VirtualCactusGroup(self.index)
        ret_val *= self
        ret_val *= other
        return ret_val

    def copy(self) -> VirtualCactusGroup:
        """
        copy everything
        """
        ret_val = VirtualCactusGroup(self.index)
        ret_val.element = [(zi.copy(),zj.copy()) for (zi,zj) in self.element]
        return ret_val

    # pylint:disable = too-many-branches,too-many-statements,too-many-locals
    def simplify(self, only_last=False, possible_pairs: Optional[List[int]] = None) -> bool:
        """
        simplify the group element which is presented as an alternating product of
        pieces from CactusGroup and SymmetricGroup
        if only_last then the only possible location for a simplification to occur is the last pair
        the optional argument of possible_pairs which specifies
        where simplifications can occur, if not only_last then this defaults to
        all the neighboring pairs
        returns whether any simplification happened to self
        """
        if only_last and len(self.element) > 1:
            possible_pairs = [len(self.element)-2]
        elif possible_pairs is None:
            possible_pairs = list(range(len(self.element)-1))
        changed = False
        cact_id = CactusGroup(self.index)
        perm_id = Permutation(self.index)
        for try_here in possible_pairs:
            try:
                cactus_part_1, sym_part_1 = self.element[try_here]
                cactus_part_2, sym_part_2 = self.element[try_here+1]
            except IndexError:
                continue
            if cactus_part_2 == cact_id and sym_part_2 == perm_id:
                continue
            if sym_part_1 == perm_id:
                changed = True
                cactus_part_1*=cactus_part_2
                self.element[try_here] = (cactus_part_1,sym_part_2)
                self.element[try_here+1] = (cact_id.copy(),perm_id.copy())
                if try_here-1 >= 0 and try_here < len(self.element):
                    possible_pairs.append(try_here-1)
                if try_here+2 < len(self.element):
                    possible_pairs.append(try_here+1)
                continue
            if cactus_part_2 == cact_id:
                changed = True
                sym_part_1*=sym_part_2
                self.element[try_here] = (cactus_part_1,sym_part_1)
                self.element[try_here+1] = (cact_id.copy(),perm_id.copy())
                if try_here-1 >= 0 and try_here < len(self.element):
                    possible_pairs.append(try_here-1)
                if try_here+2 < len(self.element):
                    possible_pairs.append(try_here+1)
                continue
            put_on_cactus_part_1 = []
            len_cact_2 = len(cactus_part_2.element)
            keep_from_here_on = 0
            for idx in range(len_cact_2):
                (z_i,z_j) = cactus_part_2.element[idx]
                z_i,z_j = min(z_i,z_j), max(z_i,z_j)
                k = sym_part_1.preserves_intervalness(z_i,z_j)
                if k is None:
                    keep_from_here_on = idx
                    break
                # maybe -k?
                put_on_cactus_part_1.append((z_i+k,z_j+k))
            else:
                keep_from_here_on = len_cact_2
            from_cactus_part_2 = CactusGroup(self.index)
            from_cactus_part_2.element = put_on_cactus_part_1
            if keep_from_here_on >= len_cact_2:
                cactus_part_1*=from_cactus_part_2
                sym_part_1*=sym_part_2
                self.element[try_here] = (cactus_part_1,sym_part_1)
                self.element[try_here+1] = (cact_id.copy(),perm_id.copy())
            else:
                cactus_part_2.element = cactus_part_2.element[keep_from_here_on:]
                cactus_part_1*=from_cactus_part_2
                self.element[try_here] = (cactus_part_1,sym_part_1)
                self.element[try_here+1] = (cactus_part_2,sym_part_2)
            changed = True
            if try_here-1 >= 0 and try_here < len(self.element):
                possible_pairs.append(try_here-1)
            if try_here+2 < len(self.element):
                possible_pairs.append(try_here+1)
        if changed:
            self.element = [(z1,z2) for (z1,z2) in self.element if (z1!=cact_id or z2!=perm_id)]
        return changed

    def __eq__(self, other) -> bool:
        """
        is self*other^-1 equal to the identity
        can say they are not equal when they are depending on simplification
        """
        if not isinstance(other, VirtualCactusGroup):
            return False
        if not self.name == other.name:
            raise TypeError("Elements of different groups")
        self.simplify()
        other.simplify()
        if len(self.element) != len(other.element):
            return False
        for ((z_i,z_j),(w_i,w_j)) in zip(self.element, other.element):
            if (z_i!=w_i) or (z_j!=w_j):
                return False
        return True

    def __str__(self):
        return "*".join((str(z_i)+"*"+str(z_j) for (z_i,z_j) in self.element))

    @staticmethod
    def random(my_n : int,my_cact_len : int,my_free_len : int) -> VirtualCactusGroup:
        """
        produce a random element of vJ_{my_n}
        my_cact_len is a bound for how many generators to use in the J_{my_n} factors
        my_free_len is a bound for how many alternating J_{my_n} and S_{my_n} factors
        """
        to_ret = VirtualCactusGroup(my_n)
        for _ in range(my_free_len):
            to_ret.element.append((CactusGroup.random(my_n,my_cact_len),Permutation.random(my_n)))
        to_ret.simplify()
        return to_ret

    def inv(self) -> VirtualCactusGroup:
        """
        return it's inverse
        """
        if len(self.element)==0:
            return self.copy()
        ret_val = VirtualCactusGroup(self.index)
        (z_i,z_j) = self.element[-1]
        ret_val *= z_j.inv()
        put_with_next = z_i.inv()
        for cur_idx in range(len(self.element)-2,-1,-1):
            (z_i,z_j) = self.element[cur_idx]
            ret_val.element.append((put_with_next,z_j.inv()))
            put_with_next = z_i.inv()
        ret_val.element.append((put_with_next,Permutation(self.index)))
        return ret_val

    def __itruediv__(self,other : VirtualCactusGroup) -> VirtualCactusGroup:
        self *= other.inv()
        return self

    def __div__(self,other : VirtualCactusGroup) -> VirtualCactusGroup:
        ret_val = VirtualCactusGroup(self.index)
        ret_val *= self
        ret_val /= other
        return ret_val
