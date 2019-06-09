import Data.List
import Data.Set

-- the k'th ECH capacity of ellipsoids with integer symplectic radii a and b
ellipsoid_capacity a b k = (sort $ toList $ fromList $ (<*> id) (\xs ys -> [a*x+b*y| x<-xs,y<-ys]) [0..k]) !! k

-- cap function
cap_function a b r = Data.List.filter (<=r) $ fmap (ellipsoid_capacity a b) [0..r]