"""
double wiring diagram
also include single wiring diagram
"""

from typing import Tuple,List,Dict,Iterator,Set,Any

from plabic import PlabicGraph, BiColor

Point = Tuple[float,float]

class WiringDiagram:
    """
    a single or double wiring diagram
    """
    def __init__(self,my_n : int, my_word : List[int]):
        if my_n<=0:
            raise ValueError("The number of strands must be a positive natural number")
        def valid_letter(letter : int) -> bool:
            """
            is (letter,letter+1) a transposition for S_{my_n}
            """
            if letter<0:
                letter = - letter
            return 0<letter<my_n

        if not all(valid_letter(z) for z in my_word):
            raise ValueError(f"The letters in the double word must be 1 to {my_n} or negative that")
        self.my_word = my_word
        self.my_n = my_n
        self.num_positive_letters = sum(1 if z>0 else 0 for z in my_word)
        self.num_negative_letters = sum(1 if z<0 else 0 for z in my_word)
        self.is_positive_order_flip = False
        self.is_negative_order_flip = False
        for _ in self.chamber_minors():
            pass

    def chamber_minors(self) -> Iterator[Tuple[Set[int],Set[int]]]:
        """
        the chamber minors of a double wiring diagram
        are the labels of the lines below each face
        here they are given as the face just to the left of
        each letter in my_word
        and then all the leftovers on the very
        right hand side of the diagram
        """
        positive_lines = list(range(1,self.my_n+1))
        negative_lines = list(range(1,self.my_n+1))
        negative_lines.reverse()
        for cur_word in self.my_word:
            abs_cur_word = cur_word if cur_word>0 else -cur_word
            flip_positive_lines = cur_word>0
            positive_lines_below = positive_lines[0:abs_cur_word]
            negative_lines_below = negative_lines[0:abs_cur_word]
            yield set(positive_lines_below),set(negative_lines_below)
            if flip_positive_lines:
                positive_lines[abs_cur_word-1],positive_lines[abs_cur_word] = \
                    positive_lines[abs_cur_word],positive_lines[abs_cur_word-1]
            else:
                negative_lines[abs_cur_word],negative_lines[abs_cur_word-1] = \
                    negative_lines[abs_cur_word-1],negative_lines[abs_cur_word]
        self.is_positive_order_flip = positive_lines_below == list(range(self.my_n,0,-1))
        self.is_negative_order_flip = negative_lines_below == list(range(self.my_n,0,-1))
        for pretend_letter in range(1,self.my_n+1):
            positive_lines_below = positive_lines[0:pretend_letter]
            negative_lines_below = negative_lines[0:pretend_letter]
            yield set(positive_lines_below),set(negative_lines_below)

    # pylint:disable=too-many-locals,too-many-branches,too-many-statements
    def to_plabic(self) -> PlabicGraph:
        """
        a plabic graph with bridges for each letter
        """
        last_on_this_strand = [[0]*self.my_n]
        for letter in self.my_word:
            if letter<0:
                letter = - letter
            cur_last_on_this_strand = last_on_this_strand[-1].copy()
            cur_last_on_this_strand[letter-1] += 1
            cur_last_on_this_strand[letter] += 1
            last_on_this_strand.append(cur_last_on_this_strand)
        very_last_numbers = last_on_this_strand[-1]
        my_init_data : Dict[str,Tuple[BiColor,List[str]]] = {}
        extra_node_props : Dict[str,Dict[str,Any]] = {}
        for idx in range(self.my_n):
            my_init_data[f"ext_left{idx+1}"] = (BiColor.RED,[f"wire{idx+1},0"])
            extra_node_props[f"ext_left{idx+1}"] = {"position" : (0,idx+1),
                                                    "my_perfect_edge":-1}
        for idx in range(self.my_n):
            my_init_data[f"ext_right{idx+1}"] =\
                (BiColor.RED,[f"wire{idx+1},{very_last_numbers[idx]-1}"])
            extra_node_props[f"ext_right{idx+1}"] = {"position" : (len(self.my_word)+1,idx+1),
                                                     "my_perfect_edge":-1}
        left_ext = [f"ext_left{idx+1}" for idx in range(self.my_n)]
        right_ext = [f"ext_right{idx+1}" for idx in range(self.my_n)]
        right_ext.reverse()
        external_init_orientation = left_ext + right_ext

        for my_idx,letter_orig in enumerate(self.my_word):
            if letter_orig<0:
                letter = - letter_orig
                bottom_color = BiColor.RED
                top_color = BiColor.GREEN
            else:
                letter = letter_orig
                top_color = BiColor.RED
                bottom_color = BiColor.GREEN
            bottom_number = last_on_this_strand[my_idx][letter-1]
            top_number = last_on_this_strand[my_idx][letter]
            bottom_name = f"wire{letter},{bottom_number}"
            top_name = f"wire{letter+1},{top_number}"
            if bottom_number==0:
                bottom_name_before = f"ext_left{letter}"
            else:
                bottom_name_before = f"wire{letter},{bottom_number-1}"
            if bottom_number==very_last_numbers[letter-1]-1:
                bottom_name_after = f"ext_right{letter}"
            else:
                bottom_name_after = f"wire{letter},{bottom_number+1}"
            if top_number==0:
                top_name_before = f"ext_left{letter+1}"
            else:
                top_name_before = f"wire{letter+1},{top_number-1}"
            if top_number==very_last_numbers[letter]-1:
                top_name_after = f"ext_right{letter+1}"
            else:
                top_name_after = f"wire{letter+1},{top_number+1}"
            my_init_data[bottom_name] = \
                (bottom_color,[bottom_name_before,top_name,bottom_name_after])
            extra_node_props[bottom_name] = {"position" : (my_idx+1,letter),
                                             "my_perfect_edge" : 1}
            my_init_data[top_name] = (top_color,[top_name_before,top_name_after,bottom_name])
            extra_node_props[top_name] = {"position" : (my_idx+1,letter+1),
                                          "my_perfect_edge" : 2}
        my_pb = PlabicGraph(my_init_data,
                 external_init_orientation,
                 {},
                 None,
                 extra_node_props)
        if not my_pb.is_bipartite():
            my_pb.remove_prop("my_perfect_edge")
        return my_pb

if __name__ == "__main__":
    double_word = [-2,1,2,-1,-2,1]
    dw = WiringDiagram(3,double_word)
    expected_chamber_minors = [
        (set([1,2]),set([2,3])),
        (set([1]),set([3])),
        (set([1,2]),set([1,3])),
        (set([2]),set([3])),
        (set([2,3]),set([1,3])),
        (set([2]),set([1])),
        (set([3]),set([1])),
        (set([2,3]),set([1,2])),
        (set([1,2,3]),set([1,2,3]))
    ]
    for expected_chamber_minor,chamber_minor in zip(expected_chamber_minors,dw.chamber_minors()):
        print(chamber_minor)
        assert chamber_minor == expected_chamber_minor
    p = dw.to_plabic()
    p.draw()
