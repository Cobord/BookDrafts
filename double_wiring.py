"""
double wiring diagram
also include single wiring diagram
"""

from typing import Tuple,List,Dict

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
            return 0<letter and letter<my_n

        if not all(valid_letter(z) for z in my_word):
            raise ValueError(f"The letters in the double word must be 1 to {my_n} or negative that")
        self.my_word = my_word
        self.my_n = my_n

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
        extra_node_props = {}
        for idx in range(self.my_n):
            my_init_data[f"ext_left{idx+1}"] = (BiColor.RED,[f"wire{idx+1},0"])
            extra_node_props[f"ext_left{idx+1}"] = {"position" : (0,idx+1)}
        for idx in range(self.my_n):
            my_init_data[f"ext_right{idx+1}"] =\
                (BiColor.RED,[f"wire{idx+1},{very_last_numbers[idx]-1}"])
            extra_node_props[f"ext_right{idx+1}"] = {"position" : (len(self.my_word)+1,idx+1)}
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
            extra_node_props[bottom_name] = {"position" : (my_idx+1,letter)}
            my_init_data[top_name] = (top_color,[top_name_before,top_name_after,bottom_name])
            extra_node_props[top_name] = {"position" : (my_idx+1,letter+1)}
        return PlabicGraph(my_init_data,
                 external_init_orientation,
                 {},
                 None,
                 extra_node_props)

if __name__ == "__main__":
    double_word = [-2,1,2,-1,-2,1]
    dw = WiringDiagram(3,double_word)
    p = dw.to_plabic()
    p.draw()
