"""
a framed little 2-disk configuration
"""
from __future__ import annotations
from typing import Callable,Tuple,List,Optional,NewType

Point = Tuple[float,float]
Radius = NewType('Radius',float)
Angle = NewType('Angle',float)

class FramedDiskConfig:
    """
    a framed little 2-disk configuration
    """

    def __init__(self,
                 internal_circles : List[Tuple[Point,Radius,Angle]],
                 outer_circle : Optional[Tuple[Point,Radius,Angle]] = None):
        if outer_circle is None:
            outer_circle = ((0.0,0.0),Radius(1.0),Angle(0.0))
        self.outer_circle = outer_circle
        self.internal_circles = internal_circles.copy()

    def find_transformation(self,which_disk : int,
                           other : FramedDiskConfig) -> Tuple[bool,Callable[[Point],Point]]:
        """
        the transformation necessary to take which_disk internal of self
        to the outer_circle of other
        """
        raise NotImplementedError(
            "Scale and rotate to get an internal disk of one to the external disk of the other")

    def draw(self,external_circle_color : str="black",
                internal_circles_color : str="black"):
        """
        draw circles
        """
        raise NotImplementedError("Draw the circles")
