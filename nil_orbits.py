from ast import Not
from typing import Dict, List, Optional, Set, Tuple

Nat = int

class YoungDiagram:

    @classmethod
    def next_partition(cls,n : int, m : Optional[int] = None) -> 'YoungDiagram':
        """Partition n with a maximum part size of m."""
        if n<0:
            raise ValueError(f"n must be non-negative")
        if m is None or m >= n:
            yield YoungDiagram([n])
        top_of_range = n-1 if (m is None or m >= n) else m
        for new_m in range(top_of_range, 0, -1):
            for p in YoungDiagram.next_partition(n-new_m, new_m):
                yield p.append(new_m)

    def __init__(self,my_partition : List[int]):
        self.my_partition = my_partition
        self.my_partition.sort(reverse=True)
        self.my_n = sum(self.my_partition)
    def append(self,new_last : int) -> 'YoungDiagram':
        return YoungDiagram(self.my_partition + [new_last])
    def n_value(self) -> int:
        return self.my_n
    def as_list(self) -> List[int]:
        return self.my_partition
    def __str__(self) -> str:
        return str(self.my_partition)

from enum import Enum, auto
class LieType(Enum):
    SL_N = auto()
    A = SL_N
    SO_2N = auto()
    D = SO_2N
    SO_2NP1 = auto()
    B = SO_2NP1
    SP_2N = auto()
    C = SP_2N

    def letter(self) -> str:
        my_letter = None
        if self is LieType.A:
            my_letter = "A"
        elif self is LieType.B:
            my_letter = "B"
        elif self is LieType.C:
            my_letter = "C"
        elif self is LieType.D:
            my_letter = "D"
        return my_letter
            

class NilpotentOrbit:
    def __init__(self,my_partition:'YoungDiagram',decorator:Optional[bool] = None,lie_type=LieType.SL_N):
        self.my_diagram = my_partition
        self.my_type = lie_type
        self.decorator = None
        my_part_temp = self.my_diagram.as_list()
        partition_n = my_partition.n_value()
        if self.my_type == LieType.D:
            if not partition_n%2==0:
                raise ValueError("The partition in type D must have been of an even natural number")
            self.lie_rank = partition_n // 2
            evens = list(filter(lambda z:z%2==0,my_part_temp))
            is_very_even = len(evens)==len(my_part_temp)
            if decorator is None and is_very_even:
                raise ValueError("Partitions with only even parts correspond to two different orbits so you must provide a boolean for the decorator to indicate which")
            elif is_very_even:
                self.decorator = decorator
            frequencies = [evens.count(even) for even in evens]
            even_parts_even_freq = all([f%2==0 for f in frequencies])
            if not even_parts_even_freq:
                raise ValueError("In type D all even parts must appear with even multiplicity")
        elif self.my_type == LieType.C:
            if not partition_n%2==0:
                raise ValueError("The partition in type C must have been of an even natural number")
            self.lie_rank = partition_n // 2
            odds = list(filter(lambda z:z%2,my_part_temp))
            frequencies = [odds.count(odd) for odd in odds]
            odd_parts_even_freq = all([f%2==0 for f in frequencies])
            if not odd_parts_even_freq:
                raise ValueError("In type C all odd parts must appear with even multiplicity")
        elif self.my_type == LieType.B:
            if not partition_n%2==1:
                raise ValueError("The partition in type B must have been of an odd natural number")
            self.lie_rank = (partition_n-1) // 2
            evens = list(filter(lambda z:z%2==0,my_part_temp))
            frequencies = [evens.count(even) for even in evens]
            even_parts_even_freq = all([f%2==0 for f in frequencies])
            if not even_parts_even_freq:
                raise ValueError("In type B all even parts must appear with even multiplicity")
        elif self.my_type == LieType.A:
           self.lie_rank = (partition_n-1)
        else:
            raise NotImplementedError("Only for one of the classical LieTypes")
        if self.lie_rank<1:
            raise ValueError("Rank must be a positive integer")

    def my_lie_alg(self) -> Tuple[LieType,Nat]:
        return (self.my_type,self.lie_rank)

    def my_dimension(self) -> Nat:
        my_part = self.my_diagram.as_list()
        sum_phat_sq = sum([z*(2*i+1) for (i,z) in enumerate(my_part)])
        dimension = 0
        num_odd_parts = sum([z%2 for z in my_part])
        if self.my_type is LieType.A:
            dimension = (self.lie_rank+1)**2 - sum_phat_sq
        elif self.my_type is LieType.D:
            dimension = 2*(self.lie_rank**2) - self.lie_rank - sum_phat_sq//2 + num_odd_parts//2
        elif self.my_type is LieType.B:
            dimension = 2*(self.lie_rank**2) + self.lie_rank - sum_phat_sq//2 + num_odd_parts//2
        elif self.my_type is LieType.C:
            dimension = 2*(self.lie_rank**2) + self.lie_rank - sum_phat_sq//2 - num_odd_parts//2
        else:
            raise ValueError("Lie type must be one of the 4 classical families")
        return dimension


    def __str__(self) -> str:
        return f"The nilpotent orbit corresponding to partition {self.my_diagram} in type {self.my_type.letter()} {self.lie_rank}"
    
    @staticmethod
    def _rank_2_n(my_type : LieType,lie_rank : Nat) -> Nat:
        if my_type is LieType.A:
            n_val = lie_rank+1
            if lie_rank<1:
                raise ValueError("Rank is too small")
        elif my_type is LieType.B:
            n_val = 2*lie_rank+1
            if lie_rank<2:
                raise ValueError("Rank is too small")
        elif my_type is LieType.C:
            n_val = 2*lie_rank
            if lie_rank<2:
                raise ValueError("Rank is too small")
        elif my_type is LieType.D:
            n_val = 2*lie_rank
            if lie_rank<3:
                raise ValueError("Rank is too small")
        else:
            raise ValueError("Lie type must be one of the 4 classical families")
        return n_val

    @classmethod
    def next_orbit(cls,my_type : LieType,lie_rank : Nat):
        # not necessarily obeying the partial order of orbit closure
        #  but it does tend to follow that ordering with the all 1's for the zero orbit at the end
        n_val = cls._rank_2_n(my_type,lie_rank)
        for diagram in YoungDiagram.next_partition(n_val):
            try:
                if my_type is LieType.D:
                    this_orbit = NilpotentOrbit(diagram,decorator=True,lie_type=my_type)
                    yield this_orbit
                    if this_orbit.decorator is not None:
                        this_orbit = NilpotentOrbit(diagram,decorator=False,lie_type=my_type)
                        yield this_orbit
                else:
                    this_orbit = NilpotentOrbit(diagram,decorator=None,lie_type=my_type)
                    yield this_orbit
            except ValueError:
                pass
    
    @classmethod
    def special_orbit(cls,my_type : LieType,lie_rank : Nat) -> Dict[str,"NilpotentOrbit"]:
        n_val = cls._rank_2_n(my_type,lie_rank)
        if my_type is LieType.A:
            prin_part = [n_val]
            subreg_part = [n_val-1,1]
            min_part = [1 for _ in range(n_val-1)]
            min_part[0] = 2
        elif my_type is LieType.B:
            prin_part = [n_val]
            subreg_part = [n_val-2,1,1]
            min_part = [1 for _ in range(n_val-2)]
            min_part[0] = 2
            min_part[1] = 2
        elif my_type is LieType.C:
            prin_part = [n_val]
            subreg_part = [n_val-2,2]
            min_part = [1 for _ in range(n_val-1)]
            min_part[0] = 2
        elif my_type is LieType.D:
            prin_part = [n_val-1,1]
            subreg_part = [n_val-3,3]
            min_part = [1 for _ in range(n_val-2)]
            min_part[0] = 2
            min_part[1] = 2
        else:
            raise ValueError("Lie type must be one of the 4 classical families")
        principal_orbit = NilpotentOrbit(YoungDiagram(prin_part),decorator=None,lie_type=my_type)
        subregular_orbit = NilpotentOrbit(YoungDiagram(subreg_part),decorator=None,lie_type=my_type)
        minimal_orbit = NilpotentOrbit(YoungDiagram(min_part),decorator=None,lie_type=my_type)
        zero_orbit = NilpotentOrbit(YoungDiagram([1 for _ in range(n_val)]),decorator=None,lie_type=my_type)
        return {"Principal":principal_orbit,"Subregular":subregular_orbit,"Minimal":minimal_orbit,"Zero":zero_orbit}
    

    def is_in_closure(self,other) -> bool:
        # is other in closure of self
        raise NotImplementedError

    def closure(self) -> List["NilpotentOrbit"]:
        raise NotImplementedError
    
    def minimal_degeneration(self) -> List["NilpotentOrbit"]:
        raise NotImplementedError