"""
nilpotent orbits
Bala-Carter
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Tuple

Nat = int


class YoungDiagram:
    """
    partitions
    """

    @classmethod
    def next_partition(cls, partition_sum: int,
                       max_part: Optional[int] = None) -> Iterator[YoungDiagram]:
        """Partition n with a maximum part size of m."""
        if partition_sum < 0:
            raise ValueError("n must be non-negative")
        if max_part is None or max_part >= partition_sum:
            yield YoungDiagram([partition_sum])
        top_of_range = partition_sum-1\
            if (max_part is None or max_part >= partition_sum)\
            else max_part
        for new_m in range(top_of_range, 0, -1):
            for p_next in YoungDiagram.next_partition(partition_sum-new_m, new_m):
                yield p_next.append(new_m)

    def __init__(self, my_partition: List[int]):
        """
        construct partition from list of parts
        """
        self.my_partition = my_partition
        self.my_partition.sort(reverse=True)
        if self.my_partition[-1]==0:
            first_zero = self.my_partition.index(0)
            self.my_partitition = self.my_partition[0:first_zero]
        self.my_n = sum(self.my_partition)

    def append(self, new_last: int) -> YoungDiagram:
        """
        a new partition which appends another part
        """
        return YoungDiagram(self.my_partition + [new_last])

    def n_value(self) -> int:
        """
        this is a partition of what n
        """
        return self.my_n

    def as_list(self) -> List[int]:
        """
        the list of parts
        """
        return self.my_partition

    def __str__(self) -> str:
        """
        for display purposes just show the list of parts
        """
        return str(self.my_partition)


class LieType(Enum):
    """
    an enumeration of the ABCD families
    can also use the Lie group name
    """
    SL_N = auto()
    A = SL_N
    SO_2N = auto()
    D = SO_2N
    SO_2NP1 = auto()
    B = SO_2NP1
    SP_2N = auto()
    C = SP_2N

    def letter(self) -> str:
        """
        the letter in the Dynkin classification
        """
        my_letter = None
        if self is LieType.A:
            my_letter = "A"
        elif self is LieType.B:
            my_letter = "B"
        elif self is LieType.C:
            my_letter = "C"
        elif self is LieType.D:
            my_letter = "D"
        else:
            raise ValueError(
                "This is not in the enum of Lie types so this should be unreachable")
        return my_letter


class NilpotentOrbit:
    """
    a nilpotent orbit in some mathfrak g
    where g is one of A,B,C,D _n
    """

    # pylint:disable = too-many-branches
    def __init__(self, my_partition: YoungDiagram,
                 decorator: Optional[bool] = None,
                 lie_type=LieType.SL_N):
        """
        in all cases the orbit is labelled by some sort of partition and possibly a boolean
        make sure the kind of partition and decorator are correct for the corresponding lie_type
        to initialize this orbit
        """
        self.my_diagram = my_partition
        self.my_type = lie_type
        self.decorator = None
        my_part_temp = self.my_diagram.as_list()
        partition_n = my_partition.n_value()
        if self.my_type == LieType.D:
            if not partition_n % 2 == 0:
                raise ValueError(
                    "The partition in type D must have been of an even natural number")
            self.lie_rank = partition_n // 2
            evens = list(filter(lambda z: z % 2 == 0, my_part_temp))
            is_very_even = len(evens) == len(my_part_temp)
            if decorator is None and is_very_even:
                msg = " ".join(["Partitions with only even parts",
                                "correspond to two different orbits",
                                "so you must provide a boolean",
                                "for the decorator to indicate which"])
                raise ValueError(msg)
            if is_very_even:
                self.decorator = decorator
            frequencies = [evens.count(even) for even in evens]
            even_parts_even_freq = all((f % 2 == 0 for f in frequencies))
            if not even_parts_even_freq:
                raise ValueError(
                    "In type D all even parts must appear with even multiplicity")
        elif self.my_type == LieType.C:
            if not partition_n % 2 == 0:
                raise ValueError(
                    "The partition in type C must have been of an even natural number")
            self.lie_rank = partition_n // 2
            odds = list(filter(lambda z: z % 2, my_part_temp))
            frequencies = [odds.count(odd) for odd in odds]
            odd_parts_even_freq = all((f % 2 == 0 for f in frequencies))
            if not odd_parts_even_freq:
                raise ValueError(
                    "In type C all odd parts must appear with even multiplicity")
        elif self.my_type == LieType.B:
            if not partition_n % 2 == 1:
                raise ValueError(
                    "The partition in type B must have been of an odd natural number")
            self.lie_rank = (partition_n-1) // 2
            evens = list(filter(lambda z: z % 2 == 0, my_part_temp))
            frequencies = [evens.count(even) for even in evens]
            even_parts_even_freq = all((f % 2 == 0 for f in frequencies))
            if not even_parts_even_freq:
                raise ValueError(
                    "In type B all even parts must appear with even multiplicity")
        elif self.my_type == LieType.A:
            self.lie_rank = partition_n-1
        else:
            raise NotImplementedError("Only for one of the classical LieTypes")
        if self.lie_rank < 1:
            raise ValueError("Rank must be a positive integer")

    def my_lie_alg(self) -> Tuple[LieType, Nat]:
        """
        which lie algebra is this orbit in
        """
        return (self.my_type, self.lie_rank)

    def my_dimension(self) -> Nat:
        """
        what is the dimension of this orbit
        """
        my_part = self.my_diagram.as_list()
        sum_phat_sq = sum((z*(2*i+1) for (i, z) in enumerate(my_part)))
        dimension = 0
        num_odd_parts = sum((z % 2 for z in my_part))
        if self.my_type is LieType.A:
            dimension = (self.lie_rank+1)**2 - sum_phat_sq
        elif self.my_type is LieType.D:
            dimension = 2*(self.lie_rank**2) - self.lie_rank - \
                sum_phat_sq//2 + num_odd_parts//2
        elif self.my_type is LieType.B:
            dimension = 2*(self.lie_rank**2) + self.lie_rank - \
                sum_phat_sq//2 + num_odd_parts//2
        elif self.my_type is LieType.C:
            dimension = 2*(self.lie_rank**2) + self.lie_rank - \
                sum_phat_sq//2 - num_odd_parts//2
        else:
            raise ValueError(
                "Lie type must be one of the 4 classical families")
        return dimension

    def __str__(self) -> str:
        """
        describe this nilpotent orbit
        """
        if self.decorator is None:
            decorator_str = ""
        elif self.decorator:
            decorator_str = "+"
        else:
            decorator_str = "-"
        return " ".join(["The nilpotent orbit corresponding",
                         f"to partition {self.my_diagram}{decorator_str}",
                         f"in type {self.my_type.letter()} {self.lie_rank}"])

    def __eq__(self, other) -> bool:
        """
        are they the same orbit
        """
        if not isinstance(other, NilpotentOrbit):
            return False
        if self.my_type != other.my_type:
            return False
        if self.lie_rank != other.lie_rank:
            return False
        if self.decorator != other.decorator:
            return False
        return self.my_diagram == other.my_diagram

    @staticmethod
    def _rank_2_n(my_type: LieType, lie_rank: Nat) -> Nat:
        """
        convert from Lie algebra rank to the dimension of the
        corresponding matrices
        """
        if my_type is LieType.A:
            # A lie_rank corresponds to SL(lie_rank+1)
            n_val = lie_rank+1
            if lie_rank < 1:
                raise ValueError("Rank is too small")
        elif my_type is LieType.B:
            # B lie_rank corresponds to SO(2*lie_rank+1)
            n_val = 2*lie_rank+1
            if lie_rank < 2:
                raise ValueError("Rank is too small")
        elif my_type is LieType.C:
            # C lie_rank corresponds to Sp(2*lie_rank)
            n_val = 2*lie_rank
            if lie_rank < 2:
                raise ValueError("Rank is too small")
        elif my_type is LieType.D:
            # D lie_rank corresponds to SO(2*lie_rank)
            n_val = 2*lie_rank
            if lie_rank < 3:
                raise ValueError("Rank is too small")
        else:
            raise ValueError(
                "Lie type must be one of the 4 classical families")
        return n_val

    @staticmethod
    def next_orbit(my_type: LieType, lie_rank: Nat,
                   max_part : Optional[int] = None) -> Iterator[NilpotentOrbit]:
        """
        generator going through all orbits of specified Lie algebra
        doesn't necessarily follow the partial order of orbit closure
        but it does tend to
        in particular the all 1's for the zero orbit is at the end
        """
        n_val = NilpotentOrbit._rank_2_n(my_type, lie_rank)
        for diagram in YoungDiagram.next_partition(n_val,max_part):
            try:
                if my_type is LieType.D:
                    this_orbit = NilpotentOrbit(
                        diagram, decorator=True, lie_type=my_type)
                    yield this_orbit
                    if this_orbit.decorator is not None:
                        this_orbit = NilpotentOrbit(
                            diagram, decorator=False, lie_type=my_type)
                        yield this_orbit
                else:
                    this_orbit = NilpotentOrbit(
                        diagram, decorator=None, lie_type=my_type)
                    yield this_orbit
            except ValueError:
                pass

    @staticmethod
    def special_orbit(my_type: LieType, lie_rank: Nat) -> Dict[str, NilpotentOrbit]:
        """
        construct all the specially named orbits of the corresponding Lie algebra
        the special names are principal, subregular, minimal and zero
        """
        n_val = NilpotentOrbit._rank_2_n(my_type, lie_rank)
        if my_type is LieType.A:
            prin_part = [n_val]
            subreg_part = [n_val-1, 1]
            min_part = [1 for _ in range(n_val-1)]
            min_part[0] = 2
        elif my_type is LieType.B:
            prin_part = [n_val]
            subreg_part = [n_val-2, 1, 1]
            min_part = [1 for _ in range(n_val-2)]
            min_part[0] = 2
            min_part[1] = 2
        elif my_type is LieType.C:
            prin_part = [n_val]
            subreg_part = [n_val-2, 2]
            min_part = [1 for _ in range(n_val-1)]
            min_part[0] = 2
        elif my_type is LieType.D:
            prin_part = [n_val-1, 1]
            subreg_part = [n_val-3, 3]
            min_part = [1 for _ in range(n_val-2)]
            min_part[0] = 2
            min_part[1] = 2
        else:
            raise ValueError(
                "Lie type must be one of the 4 classical families")
        principal_orbit = NilpotentOrbit(YoungDiagram(
            prin_part), decorator=None, lie_type=my_type)
        subregular_orbit = NilpotentOrbit(YoungDiagram(
            subreg_part), decorator=None, lie_type=my_type)
        minimal_orbit = NilpotentOrbit(YoungDiagram(
            min_part), decorator=None, lie_type=my_type)
        zero_orbit = NilpotentOrbit(YoungDiagram(
            [1 for _ in range(n_val)]), decorator=None, lie_type=my_type)
        return {"Principal": principal_orbit,
                "Subregular": subregular_orbit,
                "Minimal": minimal_orbit,
                "Zero": zero_orbit}

    def is_in_closure(self, other) -> bool:
        """
        is other in closure of self
        other <= self
        using Gerstenhaber-Hesselink theorem
        relating it to dominance order
        """
        if self.my_type != other.my_type or self.lie_rank != other.lie_rank:
            raise TypeError("The two orbits must be in the same Lie algebra")
        if self.my_type not in [LieType.A,LieType.B,LieType.C,LieType.D]:
            raise ValueError(
                "Lie type must be one of the 4 classical families")
        self_part_lengths = self.my_diagram.as_list().copy()
        other_part_lengths = other.my_diagram.as_list().copy()
        self_extend_zero_num = len(other_part_lengths) - len(self_part_lengths)
        other_extend_zero_num = -self_extend_zero_num
        self_part_lengths += [0]*self_extend_zero_num
        other_part_lengths += [0]*other_extend_zero_num
        other_up_to_k = 0
        self_up_to_k = 0
        for _k_cur,(cur_self_part,cur_other_part) in \
            enumerate(zip(self_part_lengths,other_part_lengths)):
            self_up_to_k += cur_self_part
            other_up_to_k += cur_other_part
            if other_up_to_k > self_up_to_k:
                return False
        return True

    def closure(self) -> Iterator[NilpotentOrbit]:
        """
        which orbits are in the closure of self
        """
        self_max_part = self.my_diagram.as_list()[0]
        return (orbit for orbit
                in NilpotentOrbit.next_orbit(self.my_type,self.lie_rank,self_max_part)
                if self.is_in_closure(orbit))

    def minimal_degeneration(self) -> List["NilpotentOrbit"]:
        """explain minimal degeneration"""
        raise NotImplementedError("minimal degeneration of orbit")


if __name__ == "__main__":
    for z in NilpotentOrbit.next_orbit(LieType.D,4):
        print(z)
        print([str(y) for y in z.closure()])
