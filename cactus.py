from typing import Optional,List

class CactusGroup(object):
    def __init__(self, n : int):
        self.name = f"J_{n}"
        self.index = n
        self.element = []

    def to_permutation(self) -> List[int]:
        # return list(range(1,self.index+1))
        raise NotImplementedError("Need to output the permutation")
    
    def append_generator(self,p:int,q:int):
        if (p<1 or p>self.index) or (q<1 or q>self.index) or (q<=p):
            raise ValueError(f"p,q of s_p,q must be in the range from 1 to {self.index} and p must be less than q")
        self.element.append((p,q))
        self.simplify(only_last=True)

    def __mul__(self,other : "CactusGroup"):
        if not (self.name == other.name):
            raise TypeError("Elements of different groups")
        last_of_self = len(self.element)-1
        self.element = self.element+other.element
        _ = self.simplify(only_last=False,possible_pairs=[last_of_self])
        return self

    def simplify(self,only_last=False,possible_pairs : Optional[List[int]]=None) -> bool:
        if only_last and len(self.element)>1:
            possible_pairs = [len(self.element)-2]
        elif possible_pairs is None:
            possible_pairs = range(len(self.element)-1)
        changed = False
        for try_here in possible_pairs:
            p_1, q_1 = self.element[try_here]
            p_2, q_2 = self.element[try_here+1]
            if p_1==p_2 and q_1==q_2:
                self.element = self.element[0:try_here] + self.element[try_here+2:]
                possible_pairs = [(x if x<try_here else x-2) for x in possible_pairs if x<try_here or x>try_here+1]
                possible_pairs.append(try_here-1)
                if try_here+1 < len(self.element):
                    possible_pairs.append(try_here)
            elif q_2<p_1:
                changed = True
                self.element[try_here], self.element[try_here+1] = self.element[try_here+1], self.element[try_here]
                possible_pairs.append(try_here-1)
                if try_here+2 < len(self.element):
                    possible_pairs.append(try_here+1)
            elif p_1<=p_2 and q_2<=q_1:
                changed = True
                self.element[try_here] = (p_1+q_1-q_2,p_1+q_1-p_2)
                self.element[try_here+1] = (p_1,q_1)
                possible_pairs.append(try_here-1)
                if try_here+2 < len(self.element):
                    possible_pairs.append(try_here+1)
        return changed