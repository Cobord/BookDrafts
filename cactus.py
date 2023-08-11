"""
cactus group J_n
"""

from functools import reduce
from typing import Optional, List, Tuple

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

    def __imul__(self, other: "CactusGroup"):
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
            self.element = self.element+other.element
            if last_of_self+1 < len(self.element):
                _ = self.simplify(
                    only_last=False, possible_pairs=[last_of_self])
        return self

    def __mul__(self, other: "CactusGroup"):
        ret_val = CactusGroup(self.index)
        ret_val *= self
        ret_val *= other
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
                if try_here+1 < len(self.element):
                    possible_pairs.append(try_here)
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


def perm_multiply(perm_1: List[int], perm_2: List[int]) -> List[int]:
    """
    multiply permutations 1 through n
    """
    return [perm_1[perm_2_i-1] for perm_2_i in perm_2]
