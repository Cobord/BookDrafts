import Data.List
import Data.Set

-- See https://math.berkeley.edu/~hutching/beyond.pdf for these formulas and sources for their proofs

-- the k'th ECH capacity of ellipsoids with integer symplectic radii a and b
ellipsoid_capacity a b k = (sort $ toList $ fromList $ (<*> id) (\xs ys -> [a*x+b*y| x<-xs,y<-ys]) [0..k]) !! k

-- the k'th ECH capacity of polydisks, a/b do not have to be integer in this case
polydisk_cap a b k = minimum $ (<*> id) (\xs ys -> [a*x+b*y | x <- xs, y<-ys, (x+1)*(y+1)>k+1]) [0..k]

-- cap function
cap_function a b r = Data.List.filter (<=r) $ fmap (ellipsoid_capacity a b) [0..r]
