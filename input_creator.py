# Create new locations for Phase 1
import numpy
import random
import scipy

locations = 50
homes = 20
grid_multiplier = 2 # size of grid related to number of locations
one_ways = 20

# 50.in - at most 50 locs, at most 25 TAs
# 100.in - <= 100 locs, <= 50 TAs
# 200.in - <= 200 locs, <= 100 TAs 
# make x random homes one way in/out

def create_graph(num_locs = locations, num_hom = homes, mult = grid_multiplier):
	#0 if just a node, 1 if a home
	locs = numpy.ones(num_locs)
	locs[:num_hom] = [("home_" + str(i)) for i in range(num_hom)]
	random.shuffle(locs)

	print(locs)



create_graph()