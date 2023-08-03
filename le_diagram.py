"""
Le Diagram
"""

from enum import Enum, auto
from typing import List, Optional, Tuple, Set, Iterable, Union, TypeVar, Dict, cast
import itertools

from plabic import BiColor, ExtraData, PlabicGraph

#pylint:disable=too-many-locals,too-many-statements,too-many-branches

Point = Tuple[float, float]

FillingInt = Iterable[Iterable[int]]
FillingBool = Iterable[Iterable[bool]]
FourBool = Tuple[bool, bool, bool, bool]
FourInt = Tuple[int, int, int, int]

T = TypeVar("T")
def safe_index(some_l: List[T], idx: int, default: T) -> T:
    """
    use default if would get an IndexError
    """
    return some_l[idx] if idx < len(some_l) else default

class SquareType(Enum):
    """
    the 4 possibilities for the arrow configuration
    at a 1
    """
    GAMMASHAPE = auto()
    TSHAPE = auto()
    SIDETSHAPE = auto()
    PLUSSHAPE = auto()
    JAGGEDVERTICAL = auto()
    JAGGEDHORIZONTAL = auto()


    def to_4_bools(self) -> FourBool:
        """
        where are the arrows on this kind of square
        """
        if self == SquareType.GAMMASHAPE:
            return (False,False,True,True)
        if self == SquareType.TSHAPE:
            return (False,True,True,True)
        if self == SquareType.PLUSSHAPE:
            return (True,True,True,True)
        if self == SquareType.SIDETSHAPE:
            return (True,False,True,True)
        if self == SquareType.JAGGEDVERTICAL:
            return (False,True,False,False)
        if self == SquareType.JAGGEDHORIZONTAL:
            return (True,False,False,False)
        raise ValueError(f"Unreachable, enum exhausted {self}")

def from_arrow_presence(bools : FourBool) -> SquareType:
    """
    which square type is it based on which arrows present
    """
    for sq_type in SquareType:
        if sq_type.to_4_bools() == bools:
            return sq_type
    raise ValueError(f"Impossible configuration of arrow presences {bools}")

def square_type(my_pos : Tuple[int,int],
                all_ones_data : Dict[Tuple[int,int],Tuple[FourBool,FourInt,FourBool]]
                ) -> SquareType:
    """
    the square type at my_pos based on the dictionary provided
    """
    surrounding_1s_exist,_,_ = all_ones_data[my_pos]
    arrow_data = (
        surrounding_1s_exist[0], surrounding_1s_exist[1], True, True)
    return from_arrow_presence(arrow_data)

def surrounding_names(my_pos: Tuple[int, int],
               all_ones_data : Dict[Tuple[int,int],Tuple[FourBool,FourInt,FourBool]]
                ) -> List[Tuple[str, BiColor, List[str],Tuple[float,float]]]:
    """
    the name of the vertex/vertices at my_pos, it's color/colors
    the neighbors in clockwise order, and the position(s) of the
    vertex/vertices at my_pos
    basically the coordinates of my_pos moved from indexing into the tableau
    to where they would be drawn in English convention
    """
    (surrounding_1s_exist,surrounding_1s_locs,_) = all_ones_data[my_pos]

    if surrounding_1s_exist[0]:
        above_loc = (surrounding_1s_locs[0],my_pos[1])
        above_square_type = square_type(above_loc,all_ones_data)
        if above_square_type == SquareType.PLUSSHAPE:
            above_name = f"int{above_loc[0]},{above_loc[1]}_bl"
        else:
            above_name = f"int{above_loc[0]},{above_loc[1]}"
    else:
        above_name = ""

    if surrounding_1s_exist[1]:
        left_loc = (my_pos[0],surrounding_1s_locs[1])
        left_square_type = square_type(left_loc,all_ones_data)
        if left_square_type == SquareType.PLUSSHAPE:
            left_name = f"int{left_loc[0]},{left_loc[1]}_tr"
        else:
            left_name = f"int{left_loc[0]},{left_loc[1]}"
    else:
        left_name = ""

    if surrounding_1s_exist[2]:
        below_loc = (surrounding_1s_locs[2],my_pos[1])
        below_square_type = square_type(below_loc,all_ones_data)
        if below_square_type == SquareType.PLUSSHAPE:
            below_name = f"int{below_loc[0]},{below_loc[1]}_tr"
        else:
            below_name = f"int{below_loc[0]},{below_loc[1]}"
    else:
        below_loc = (surrounding_1s_locs[2],my_pos[1])
        below_name = f"bdry{below_loc[0]},{below_loc[1]}_h"

    if surrounding_1s_exist[3]:
        right_loc = (my_pos[0],surrounding_1s_locs[3])
        right_square_type = square_type(right_loc,all_ones_data)
        if right_square_type == SquareType.PLUSSHAPE:
            right_name = f"int{right_loc[0]},{right_loc[1]}_bl"
        else:
            right_name = f"int{right_loc[0]},{right_loc[1]}"
    else:
        right_loc = (my_pos[0],surrounding_1s_locs[3])
        right_name = f"bdry{right_loc[0]},{right_loc[1]}_v"

    my_square = square_type(my_pos, all_ones_data)
    my_name = f"int{my_pos[0]},{my_pos[1]}"

    if my_square == SquareType.GAMMASHAPE:
        return [(my_name,BiColor.RED,[right_name,below_name],(0.0,0.0))]
    if my_square == SquareType.TSHAPE:
        return [(my_name,BiColor.GREEN,[right_name,below_name,left_name],(0.0,0.0))]
    if my_square == SquareType.PLUSSHAPE:
        my_name_red = f"{my_name}_tr"
        my_name_green = f"{my_name}_bl"
        my_red = (my_name_red,BiColor.RED,[above_name,right_name,my_name_green],(0.1,0.1))
        my_green = (my_name_green,BiColor.GREEN,[my_name_red,below_name,left_name],(-0.1,-0.1))
        return [my_red,my_green]
    if my_square == SquareType.SIDETSHAPE:
        return [(my_name,BiColor.RED,[right_name,below_name,above_name],(0.0,0.0))]
    raise ValueError("Should be unreachable, cases for 1 existing enum exhausted")

class LeDiagram:
    """
    A Young diagram
    with 0,1 filling
    satisfing the
    backwards L condition
    for the position of 1's
    when drawn in English convention
    """
    def __init__(self,pre_filling : Union[FillingInt,FillingBool]):
        """
        give the filling along each row
        left to right and top to bottom
        """
        filling : List[List[bool]] = \
            list(map(lambda cur_row : list(map(lambda z:z>0,cur_row)),pre_filling))
        my_part_lengths = [len(z) for z in filling]
        if sum(my_part_lengths)==0:
            raise ValueError("Expect the shape to be a partition of a positive number")
        for (row_len_a,row_len_b) in zip(my_part_lengths,my_part_lengths[1:]):
            if row_len_a<row_len_b:
                raise ValueError(f"{my_part_lengths} did not describe a Young Diagram")
        self.width = my_part_lengths[0]
        self.height = len(my_part_lengths)
        self.my_diagram = my_part_lengths
        locations_1s : Set[Tuple[int,int]] = set()
        for row_idx, cur_row in enumerate(filling):
            for col_idx, cur_filling in enumerate(cur_row):
                if cur_filling:
                    locations_1s.add((row_idx,col_idx))
        for ((i_1,j_1),(i_2,j_2)) in itertools.combinations(locations_1s,2):
            if i_1<i_2 and j_1>j_2:
                if not filling[i_2][j_1]:
                    raise ValueError(f"{(i_1,j_1)} and {(i_2,j_2)} were 1, but {(i_2,j_1)} was not")
        ones_data : Dict[Tuple[int,int],Tuple[FourBool,FourInt,FourBool]] = {}

        self.column_heights : List[Optional[int]] = [None]*self.width
        self.filling = filling

        for (cur_row_idx,cur_col_idx) in locations_1s:

            all_above = \
                list(filling[cur_row_idx-prev_row][cur_col_idx]
                     for prev_row in range(1,cur_row_idx+1))
            has_1_above = any(all_above)
            if has_1_above:
                loc_1_above = all_above.index(True)
                loc_1_above = cur_row_idx - loc_1_above - 1
                above_off_d = False
            else:
                loc_1_above = -1
                above_off_d = True

            all_below = list(safe_index(filling[next_row],cur_col_idx,False)
                                 for next_row in range(cur_row_idx+1,len(filling)))
            has_1_below = any(all_below)
            if has_1_below:
                loc_1_below = all_below.index(True)
                loc_1_below = cur_row_idx + loc_1_below + 1
                below_off_d = False
            else:
                below_off_d = True
                loc_1_below = cast(int,self.column_height(cur_col_idx))

            all_right = list(filling[cur_row_idx][next_col]
                              for next_col in range(cur_col_idx+1,len(filling[cur_row_idx])))
            has_1_right = any(all_right)
            if has_1_right:
                loc_1_right = all_right.index(True)
                loc_1_right = loc_1_right + cur_col_idx+1
                right_off_d = False
            else:
                loc_1_right = len(filling[cur_row_idx])
                right_off_d = True

            all_left = \
                list(filling[cur_row_idx][cur_col_idx-prev_col]
                     for prev_col in range(1,cur_col_idx+1))
            has_1_left = any(all_left)
            if has_1_left:
                loc_1_left = all_left.index(True)
                loc_1_left = cur_col_idx - loc_1_left - 1
                left_off_d = False
            else:
                left_off_d = True
                loc_1_left = -1

            ones_data[(cur_row_idx,cur_col_idx)] =\
                ((has_1_above,has_1_left,has_1_below,has_1_right),
                 (loc_1_above,loc_1_left,loc_1_below,loc_1_right),
                 (above_off_d,left_off_d,below_off_d,right_off_d))
        self.ones_data = ones_data

    def column_height(self,which_col : int) -> Optional[int]:
        """
        how many boxes in this column, None which_col is <0 or >=self.width
        """
        if which_col<0 or which_col>=self.width:
            return None
        cached_ht = self.column_heights[which_col]
        if cached_ht is not None:
            return cached_ht
        my_ht = 0
        for my_ht in range(self.height+1):
            try:
                self.filling[my_ht][which_col]
            except IndexError:
                self.column_heights[which_col] = my_ht
                return my_ht
        return None

    def to_plabic(self) -> PlabicGraph:
        """
        a plabic graph with bridges for each letter
        """

        my_init_data : Dict[str, Tuple[BiColor, List[str]]] = {}
        extra_node_props: Dict[str, ExtraData] = {}

        # on isolated boundary vertices, there is a lollipop
        # what color is the lollipop to make
        # ones for the vertical edges of the jagged path be sources
        # ones for the horizontal edges of the jagged path be sinks
        color_to_make_source = BiColor.RED
        color_to_make_sink = BiColor.GREEN

        for one_pos in self.ones_data:
            my_names_and_neighbors = \
                surrounding_names(one_pos,self.ones_data)
            for my_name,my_color,clockwise_neighbors,pos_shift in my_names_and_neighbors:
                my_init_data[my_name] = (my_color,clockwise_neighbors)
                extra_node_props[my_name] = {"position" :
                                             (one_pos[1]+pos_shift[0],-one_pos[0]+pos_shift[1])}

        cur_box = (0,self.width)
        cur_bdry_type = SquareType.JAGGEDVERTICAL
        external_init_orientation : List[str] = []
        empty_bool_list : List[bool] = []
        for _ in range(self.width+self.height):
            this_bdry_non_lollipop = True
            if cur_bdry_type == SquareType.JAGGEDVERTICAL:
                my_name = f"bdry{cur_box[0]},{cur_box[1]}_v"
                cur_list = self.filling[cur_box[0]].copy()
                cur_list.reverse()
                try:
                    pos_last_1 = cur_box[1] - cur_list.index(True) - 1
                except ValueError:
                    this_bdry_non_lollipop = False
                if this_bdry_non_lollipop:
                    last_in_this_row = f"int{cur_box[0]},{pos_last_1}"
                    last_in_this_row_type = square_type((cur_box[0],pos_last_1),self.ones_data)
                    if last_in_this_row_type == SquareType.PLUSSHAPE:
                        last_in_this_row = f"{last_in_this_row}_tr"
                    my_init_data[my_name] = (BiColor.RED,[last_in_this_row])
                    my_position = (cur_box[1]-0.5,-cur_box[0]+0.0)
                    extra_node_props[my_name] = {"position" : my_position}
                else:
                    my_lollipop_name = f"{my_name}_lollipop"
                    my_init_data[my_name] = (BiColor.RED,[my_lollipop_name])
                    my_position = (cur_box[1]-0.5,-cur_box[0]+0.0)
                    extra_node_props[my_name] = {"position" : my_position}
                    my_init_data[my_lollipop_name] = (color_to_make_source,[my_name])
                    my_position = (cur_box[1]-0.7,-cur_box[0]+0.0)
                    extra_node_props[my_lollipop_name] = {"position" : my_position}
                if len(safe_index(self.filling,cur_box[0]+1,empty_bool_list))==cur_box[1]:
                    cur_box = (cur_box[0]+1,cur_box[1])
                    cur_bdry_type = SquareType.JAGGEDVERTICAL
                else:
                    cur_box = (cur_box[0]+1,cur_box[1]-1)
                    cur_bdry_type = SquareType.JAGGEDHORIZONTAL
            else:
                my_name = f"bdry{cur_box[0]},{cur_box[1]}_h"
                pos_last_1 = cur_box[0]-1
                while pos_last_1>=0 and not self.filling[pos_last_1][cur_box[1]]:
                    pos_last_1 -= 1
                if pos_last_1 == -1:
                    this_bdry_non_lollipop = False
                if this_bdry_non_lollipop:
                    last_in_this_col = f"int{pos_last_1},{cur_box[1]}"
                    last_in_this_col_type = square_type((pos_last_1,cur_box[1]),self.ones_data)
                    if last_in_this_col_type == SquareType.PLUSSHAPE:
                        last_in_this_col = f"{last_in_this_col}_bl"
                    my_init_data[my_name] = (BiColor.RED,[last_in_this_col])
                    my_position = (cur_box[1]+0.0,-cur_box[0]+0.5)
                    extra_node_props[my_name] = {"position" : my_position}
                else:
                    my_lollipop_name = f"{my_name}_lollipop"
                    my_init_data[my_name] = (BiColor.RED,[my_lollipop_name])
                    my_position = (cur_box[1]+0.0,-cur_box[0]+0.5)
                    extra_node_props[my_name] = {"position" : my_position}
                    my_init_data[my_lollipop_name] = (color_to_make_sink,[my_name])
                    my_position = (cur_box[1]+0.0,-cur_box[0]+0.7)
                    extra_node_props[my_lollipop_name] = {"position" : my_position}
                if self.column_height(cur_box[1]-1) == cur_box[0]:
                    cur_box = (cur_box[0],cur_box[1]-1)
                    cur_bdry_type = SquareType.JAGGEDHORIZONTAL
                else:
                    cur_box = (cur_box[0],cur_box[1])
                    cur_bdry_type = SquareType.JAGGEDVERTICAL
            external_init_orientation.append(my_name)

        return PlabicGraph(my_init_data,
                 external_init_orientation,
                 {},
                 [],
                 extra_node_props)

    def __str__(self) -> str:
        return "\n".join([
            " ".join(["1" if z else "0" for z in cur_row])
            for cur_row in self.filling])

if __name__ == "__main__":

    try:
        my_Le = LeDiagram([])
    except ValueError as e:
        assert str(e)=="Expect the shape to be a partition of a positive number"
    try:
        my_Le = LeDiagram([[],[],[]])
    except ValueError as e:
        assert str(e)=="Expect the shape to be a partition of a positive number"

    # Example from
    # https://doi.org/10.1016/j.jcta.2018.06.001
    my_diagram = [[0,1,0,1,0],[1,1,0,1],[0,0],[0,1]]
    my_Le = LeDiagram(my_diagram)
    print(my_Le)
    print(my_Le.ones_data)
    print(my_Le.column_height(0))
    p = my_Le.to_plabic()
    p.draw(show_node_names=False)

    # case of n by k rectangle with all 1's
    # have particularly nice face quivers
    k,n = 5,4
    my_diagram = [[1]*k]*n
    my_Le = LeDiagram(my_diagram)
    p = my_Le.to_plabic()
    p.draw(show_node_names=False)
