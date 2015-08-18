import my_graph_helpers as mgh
import my_graph as mg
import spatial_plotting as sp

from matplotlib import pyplot as plt

import numpy as np
import networkx as nx
import itertools as it

import pandas as pd
import json




def define_clusters(myG):
    """Returns a list of connected components (edgewise consistent with main graph)
    that represent the new trees built into the graph.
       """
    
    decimate = myG.copy()
    
    for e in decimate.myedges():
        if e not in new_roads:
            decimate.remove_myedges_from([e])
        for n in decimate.G.nodes():
            if decimate.G[n] == {}:
                decimate.G.remove_node(n)

    clusters = nx.connected_components(decimate.G)
    return clusters

def cluster_to_cluster_path(graph,cluster0,cluster1):
    """ Finds the shortest path between two clusters. """
    fakenode1 = mg.MyNode([-1,1])
    fakenode2 = mg.MyNode([1,-1])

    graph.add_node(fakenode1)
    graph.add_node(fakenode2)

    for n in cluster0:
        mgh._fake_edge(graph,fakenode1,n)

    for n in cluster1:
        mgh._fake_edge(graph, fakenode2, n)


    path = nx.shortest_path(graph.G, fakenode1, fakenode2, "weight")
    length = nx.shortest_path_length(graph.G, fakenode1, fakenode2, "weight")
    
    mypath = mgh.ptup_to_mypath(graph,path[1:-1])
    
    graph.G.remove_nodes_from([fakenode1,fakenode2])
    
    return path[1:-1], length

def make_row_clusters(graph,c0,c1):
    """find all the important results about travel between two given clusters.
    Returns pd data series   """
    path, length = cluster_to_cluster_path(graph,c0,c1)
    copy = graph.copy()
    new_roads = build_path(copy,path)
    matrix, mean_travel = mgh.shortest_path_p2p_matrix(copy)
    # mgh.plot_cluster_mat(matrix, matrix, "testtitle", plot_dendro=True)
    y = pd.Series({'cluster1':c0, 'cluster2':c1, 'path':path, 'pathlength':length, 
                   'meantravel':mean_travel, 'tcmatrix':matrix})
    return y

def exhaustive_by_clusters(graph):
    """searches all pairs of clusters for path and change in Tbar. writes and returns results in dataframe"""
    clusters = define_clusters(graph)
    df = pd.DataFrame(columns=['cluster1','cluster2','path','pathlength','meantravel', 'tcmatrix'])
    counter = 0

    for (c0,c1) in it.combinations(clusters,2):
        y = make_row_clusters(graph,c0,c1)
        df.loc[counter] = y
        counter += 1
        df.to_pickle(graph.name+"travel_cost_df")
        print counter

    df['ratio'] = df.pathlength/(basetc - df.meantravel)
    
    return df

def find_specialparcel_nodes(graph, network_matrix, geometric_matrix):
    """Finds the special parcel based on the minimum ratio of
    network travel distance to geometric travel distance  """
    
    nrs = [sum(row) for row in network_matrix]
    grs = [sum(row) for row in geometric_matrix]
    ratio = [grs[i]/nrs[i] for i in range(0, len(nrs))]
    specialindex = ratio.index(min(ratio))
    specialparcel = graph.inner_facelist[specialindex]
    
    startnodes = [n for n in specialparcel.nodes if n.road]
    if len(startnodes) == 0:
        raise ValueError('special parcel should have road nodes!')
    
    potential_road_nodes = [n for n in graph.road_nodes if len(graph.G[n]) >= 3]
    
    return specialindex, specialparcel, startnodes, potential_road_nodes
    
def find_pairs(graph, startnodes, potential_road_nodes):

    """ finds all pairs of nodes, and the ratio of geometric to network
     travel distance between them for a given set of starting
     and ending nodes (based on a particular special parcel)
     """
    
    pairs = {}
    roads_only = graph.copy()
    etup_drop = roads_only.find_interior_edges()
    roads_only.G.remove_edges_from(etup_drop)
    mgh.__road_connections_through_culdesac(roads_only)
    

    for sn in startnodes:
        for prn in potential_road_nodes:
            if sn is not prn:
                geometric_distance = mgh.distance(sn,prn)
                network_distance = nx.shortest_path_length(roads_only.G, sn, prn, "weight")
                ratio = geometric_distance/network_distance
                pairs[(sn,prn)]=ratio
                
    return pairs

def build_path(graph,pairs):
    """builds a path based on the minimum ratio of geometric to travel distance.
    """
    target = min(pairs, key=pairs.get)
    path = nx.shortest_path(graph.G,target[0],target[1])
    length = nx.shortest_path_length(graph.G,target[0],target[1], 'weight')
    
    mgh.build_path(graph,target[0],target[1])
    return path, length


def one_bisection(graph, Tminus1, geometric_matrix, plotting_data, verbose=False):
    """plotting_data = [master, title, fbase, name, T0, dmax]   """

    [master, title, fbase, name, T0, dmax] = plotting_data

    sindex, sparcel, startn, potential_rn = find_specialparcel_nodes(graph,Tminus1,
                                                                     geometric_matrix)
    if verbose is True:
        print "special parcel index is {}".format(sindex)
        
    pairs = find_pairs(graph, startn, potential_rn)
    path, length = build_path(graph,pairs)
    resultT, resultTbar = mgh.shortest_path_p2p_matrix(graph)
    if verbose is True:
        print "Average resulting travel cost is {}".format(resultTbar)
        print "new path length is {0:.2f}".format(length)

    y = pd.Series({ 'resultT':resultT, 'resultTbar':resultTbar, 'pathlength':length,
                    'specialindex':sindex, 'title': title})

    #plot results
    
    sp.plot_roads(graph, master = master, title = title, old_node_size=10, old_road_width = 4)
    plt.savefig(fbase+name+title+"map.pdf", format = 'pdf')

    sp.plot_cluster_mat(T0, resultT, title, dmax=dmax, plot_dendro=False)
    plt.savefig(fbase+name+title+"T.pdf", format = 'pdf')

    np.savetxt(fbase+name+title+"T.csv", resultT, delimiter=",")
     
    return y

def geometric_distance_matrix(myG):
    """option if full is false, removes all interior edges, so that travel
    occurs only along a road.  If full is true, keeps interior edges.  If
    travel cost is true, increases weight of non-road edges by a factor of ten.
    Base case is defaults of false and false."""

    copy = myG.copy()

    n = len(copy.inner_facelist)
    tcmat = np.zeros((n,n))

    for (p0,p1) in it.combinations(copy.inner_facelist,2):
        p0index = copy.inner_facelist.index(p0)
        p1index = copy.inner_facelist.index(p1)
        
        distance = mgh.distance(p0.centroid,p1.centroid)

        tcmat[p0index][p1index]=distance
        tcmat[p1index][p0index]=distance
        
    meantravel = tcmat.mean()

    return tcmat, meantravel


def check_is_are(list1, list2):
    counter = 0
    list1problems = []
    list2problems = []
    for n1 in list1:
        for n2 in list2:
            result = mgh.test_nodes(n1,n2)
            if result is False:
                list1problems.append(n1)
                list2problems.append(n2)
                counter += 1
    
    print "{} pairs of problem nodes".format(counter)
    return list1problems, list2problems

def test_graph_is(graph):
    pnl = []
    enl = []
    for e in graph.myedges():
        enl.append(e.nodes[0])
        enl.append(e.nodes[1])
        
    for p in graph.inner_facelist:
        for n in p.nodes:
            pnl.append(n)
            
    gnl = graph.G.nodes()
    result = mgh.test_nodes(enl,gnl)
    if result == 0:
        print "all equal nodes are is for edges"
        
    result = mgh.test_nodes(pnl,gnl)
    if result == 0:
        print "all equal nodes are is for parcels"
    else:
        print "all equal nodes are NOT is for parcels"
        
    return result

