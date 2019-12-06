import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils

from student_utils import *

"""
======================================================================
  Complete the following function.
======================================================================
"""

import numpy as np
import networkx as nx
import student_utils as s_utils
import utils

#------------------

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    # Make networkx graph G
    #G = make_graph(adjacency_matrix)
    G, message = s_utils.adjacency_matrix_to_graph(adjacency_matrix)
    #print("G:", G.edges())
    
    # Get average distance in between locations
    adj = np.array(adjacency_matrix)
    adj[adj == "x"] = 0
    adj = adj.astype(float)
    #print(adj[0][0])
    avg_dist = np.mean(adj)
    
    # Get indices of homes
    home_indices = [list_of_locations.index(i) for i in list_of_homes]
    location_indices = range(0, len(list_of_locations))
    
    starting_index = list_of_locations.index(starting_car_location)
    #print("start index:", starting_index)
    
    # Get adjacency dictionary of distances for each location
    adjacencies = make_dictionary(adjacency_matrix, location_indices)
    #print("adjacencies:", adjacencies)
    
    # Get nodes
    nodes = make_nodes(adjacencies, location_indices, home_indices, starting_index, avg_dist)
    node_roots = list(nodes.keys())
    #print("node_roots:", node_roots)
    
    # Create graph of just nodes
    node_paths, node_G = shortest_paths(G, list_of_locations, node_roots)
    #print("node_G edges:", node_G.edges())
    #print("edge (5,6)", node_G.get_edge_data(5, 6,default=0) )
    #print("edge (5,8)", node_G.get_edge_data(5, 8,default=0) )
    #print(node_G.nodes())
        
    # TSP on node_G
    #path = tsp_solver(node_G, starting_index)
    tsp_path = christofides(node_G, starting_index)
    #print("node_path:", tsp_path)
    
    # Output path for driver IN LOCATIONS
    output_path = make_path(list_of_locations, node_paths, tsp_path, starting_index)
    #print("output path:", output_path)
    
    #print("tsp_path:", tsp_path, "\n path:", output_path, "\n homes:", home_indices, "\n nodes:", nodes)
    
    # Drop off points and TAs dropped
    #dropoff_mapping = dicitonary of {dropoff_loc: [list of TAs dropped off], ...} 
    dropoff_mapping = dropoffs(output_path, nodes, tsp_path, home_indices, list_of_locations)
    
    # Create output file
    #print("output:", output_path, dropoff_mapping)
    return output_path, dropoff_mapping

# Helpers --------------------------------

# Returns adjacency dictionary of distances for each location {location: [distances to every location], ...}
def make_dictionary(adjacency_matrix, location_indices):
    #Create dictionary for every location
    adjacencies = {}
    
    #Create dictionary of adjacencent locations for every location
    # adjacencies = {"loc" : [distance to every other loc], ...}
    # If distance == "x" -> None
    for i in location_indices:
        adj = [None if j == "x" else j for j in adjacency_matrix[i]]
        adjacencies[i] = adj
    
    return adjacencies

# Returns dictionary of {node_home: [homes belonging to node], ...}
def make_nodes(adjacencies, locations, home_indices, starting_index, avg_dist):    
    #limit = 12000
    limit = avg_dist #maximum distance away from node's base (average of all distances in adjacency)
    nodes = {} #create a node around every location
    
    #Create nodes for every home
    for loc in locations:
        if loc in home_indices:
            nodes[loc] = list()
            nodes[loc].append(loc) #Start every home's node with itself
        else:
            nodes[loc] = list()
        
        for index in home_indices:
            distance = adjacencies[loc][index]
            if (distance != None) and (distance < limit):
                #append other home that is within limit to the node starting at that home
                #current = nodes[loc]
                nodes[loc].append(index) #returns None    
    
    #Clean up node dictionary to only contain largest nodes ------------
    deleted_nodes = nodes.copy()
    homes_represented = home_indices
    nodes_to_keep = list()
    
    #for node in node.keys():
    
    while homes_represented:        
        v = list(deleted_nodes.values())
        k = list(deleted_nodes.keys())
        biggest_node = k[v.index(max(v, key=len))]
        
        #remove homes that are already included in list
        #print(len(homes_represented))
        homes_represented = [x for x in homes_represented if x not in nodes[biggest_node]]
        
        deleted_nodes.pop(biggest_node, None)
        
        for home in nodes[biggest_node]:
            deleted_nodes.pop(home, None)
            #print("deleted_nodes:", deleted_nodes)
        
        nodes_to_keep.append(biggest_node)
    
    #print(nodes_to_keep)
    
    if starting_index not in nodes_to_keep:
        nodes_to_keep.append(starting_index)
    
    return {key: nodes[key] for key in nodes_to_keep}

#returns dictionary of shortest paths between nodes {(node_1, node_2): [list of path], ...} 
#    for outputting driver path
#returns new graph of just nodes and their associated distances
#    for TSP solving with dwave
def shortest_paths(G, list_of_locations, node_roots):
    node_paths = {}
    #node_distances = {}
    
    #Make new graph of nodes
    node_G = nx.Graph()
    
    for node in node_roots:
        node_G.add_node(node)
    
    #Get shortest path between every node and the distance of that path
    for node_s in node_roots:
        node_paths[node_s] = list()
        for node_t in node_roots:
            #index_s = list_of_locations.index(node_s)
            #index_t = list_of_locations.index(node_t)
            if (node_s != node_t):
                path = nx.shortest_path(G, source = node_s, target = node_t, weight = "weight")
                node_paths[(node_s, node_t)] = path

                path_weight = nx.shortest_path_length(G, source= node_s, target= node_t, weight= "weight")
                node_G.add_edge(node_s, node_t, weight= path_weight)
        
    return node_paths, node_G

# Christofides TSP Solver  
def christofides(G, starting_node):
    optimal_path = list()
    
    MST = nx.minimum_spanning_tree(G, weight='weight') # generates MST of graph G, using Prim's algo
    odd_vertices = [] #list containing vertices with odd degree
    
    for i in MST.nodes():
        if MST.degree(i)%2 != 0: 
            odd_vertices.append(i) #if the degree of the vertex is odd, then append it to odd_vertices list
            
    minimumWeightedMatching(MST, G, odd_vertices) #adds minimum weight matching edges to MST
    
    # now MST has the Eulerian circuit
    start = starting_node
    visited = {node: False for node in MST.nodes()}
    
    # finds the hamiltonian circuit (skips repeated vertices)
    curr = start
    visited[curr] = True
    optimal_path.append(curr)
    
    next = None
    
    for nd in MST.neighbors(curr):
        if visited[nd] == False or nd == start:
            next = nd
            break
            
    if next == None:
        return [start]
      
    while next != start:
        visited[next]=True
        optimal_path.append(next)
        # finding the shortest Eulerian path from MST
        curr = next
        for nd in MST.neighbors(curr):
            if visited[nd] == False:
                next = nd
                break
        if next == curr:
            for nd in G.neighbors(curr):
                if visited[nd] == False:
                    next = nd
                    break
        if next == curr:
            next = start
    
    optimal_path.append(next)
    
    return optimal_path

#utility function that adds minimum weight matching edges to MST
def minimumWeightedMatching(MST, G, odd_vertices):
    while odd_vertices:
        y = odd_vertices.pop()
        weight = float("inf")
        x = 1

        closest = 0
        for x in odd_vertices:
            if G[y][x]['weight'] < weight :
                weight = G[y][x]['weight']
                closest = x

        MST.add_edge(y, closest, weight = weight)
        odd_vertices.remove(closest)

# Create output path for driver
def make_path(list_of_locations, node_paths, tsp_path, starting_index):
    output_path = list()
    p_prev = starting_index
    output_path.append(starting_index)
    
    #print("node paths:", node_paths)
    
    for p_curr in tsp_path[1:]:
        if node_paths[(p_prev, p_curr)][1]:
            path = node_paths[(p_prev, p_curr)][1:]
            output_path.extend(path)
        else:
            path = node_paths[(p_curr, p_prev)][::-1] 
            path = path[1:]
            output_path.extend(path)
            
        p_prev = p_curr
    
    return output_path

# Create dictionary for drivers and paths

"""
dropped_TAs = Create dictionary of home_indices for whether or not they've been dropped off
dropoff_mapping = Create dictionary of every location on output_path and accompanying list for TAs dropped there

Iterate through output_path:
    If path stop matches home, drop off TA
    Mark in dropped_TAs
    Add TA to dropoff_mapping for that location (in terms of INDEX)
    
Iterate through tsp_path:
    For every node, drop off associated TAs if they haven't been dropped off yet
    Mark in dropped_TAs
    Add TA to dropoff_mapping for that node (in terms of INDEX)
    
Sort dropoff_mapping keys in order they are reached in output_path
return dropoff_mapping
"""
def dropoffs(output_path, nodes, tsp_path, home_indices, list_of_locations):
    dropped_TAs = {}
    dropoff_mapping = {}
    
    for home in home_indices:
        dropped_TAs[home] = False
    
    for loc in output_path:
        if loc in home_indices:
            if not dropped_TAs[loc]:
                dropped_TAs[loc] = True
                
                if not loc in dropoff_mapping.keys():
                    dropoff_mapping[loc] = list()
                dropoff_mapping[loc].append(loc)
        
    for loc in output_path:
        if loc in tsp_path:
            for ta in nodes[loc]:
                if not dropped_TAs[ta]:
                    dropped_TAs[ta] = True
                    
                    if not loc in dropoff_mapping.keys():
                        dropoff_mapping[loc] = list()
                    dropoff_mapping[loc].append(ta)
    
    for ta in list(dropped_TAs.keys()):
        if not dropped_TAs[ta]:
            raise ValueError("This TA hasn't been dropped off: ", ta)
    
    #sort dropoffs?
    return dropoff_mapping


"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        if (input_file[8] == "_"):
            #------------
            #if (int(input_file[7:9]) % 5 == 0):
            #print(input_file)
            #------------

            solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
