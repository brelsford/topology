import numpy as np
import shapefile
from matplotlib import pyplot as plt
import networkx as nx
import random
import itertools
import operator
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.plotly as py
from plotly.graph_objs import *

import my_graph as mg


""" This file includes a bunch of helper functions for my_graph.py.
<<<<<<< HEAD
There are a bunch of basic spatial geometery functions, some greedy search
probablilty functions,
=======

There are a bunch of basic spatial geometery functions,

some greedy search probablilty functions,
>>>>>>> master

ways to set up and determine the shortest paths from parcel to a road

the code that exists on optimization problem 2: thinking about how to build in
additional connectivity beyond just minimum access, as well as plotting the
associated matrices

code for creating a mygraph object from a shapefile or a list of myfaces
(used for weak dual calculations)

a couple of test graphs- testGraph, (more or less lollipopo shaped) and
testGraphLattice which is a lattice.

   """

#############################
# BASIC MATH AND GEOMETRY FUNCTIONS
#############################


# myG geometry functions
def distance(mynode0, mynode1):
    return np.sqrt(distance_squared(mynode0, mynode1))


def distance_squared(mynode0, mynode1):
    return (mynode0.x-mynode1.x)**2+(mynode0.y-mynode1.y)**2


def sq_distance_point_to_segment(target, myedge):
    """returns the square of the minimum distance between mynode
    target and myedge.   """
    n1 = myedge.nodes[0]
    n2 = myedge.nodes[1]

    if myedge.length == 0:
        sq_dist = distance_squared(target, n1)
    elif target == n1 or target == n2:
        sq_dist = 0
    else:
        px = float(n2.x - n1.x)
        py = float(n2.y - n1.y)
        u = float((target.x - n1.x)*px + (target.y - n1.y)*py)/(px*px + py*py)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        x = n1.x + u*px
        y = n1.y + u*py

        dx = x - target.x
        dy = y - target.y

        sq_dist = (dx * dx + dy * dy)
    return sq_dist


def intersect(e1, e2):
    """ returns true if myedges e1 and e2 intersect """
    # fails for lines that perfectly overlap.
    def ccw(a, b, c):
        return (c.y-a.y)*(b.x-a.x) > (b.y-a.y)*(c.x-a.x)

    a = e1.nodes[0]
    b = e1.nodes[1]
    c = e2.nodes[0]
    d = e2.nodes[1]

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def are_parallel(e1, e2):
    """ returns true if myedges e1 and e2 are parallel """
    a = e1.nodes[0]
    b = e1.nodes[1]
    c = e2.nodes[0]
    d = e2.nodes[1]

    # check if parallel; handling divide by zero errors
    if a.x == b.x and c.x == d.x:  # check if both segments are flat
        parallel = True
    # if one is flat and other is not
    elif (a.x - b.x)*(c.x - d.x) == 0 and (a.x - b.x) + (c.x - d.x) != 0:
        parallel = False
    # if neither segment is flat and slopes are equal
    elif (a.y-b.y)/(a.x-b.x) == (c.y-d.y)/(c.x-d.x):
        parallel = True
    # n either segment is flat, slopes are not equal
    else:
        parallel = False
    return parallel


def segment_distance_sq(e1, e2):
    """returns the square of the minimum distance between myedges e1 and e2."""
    # check different
    if e1 == e2:
        sq_distance = 0
    # check parallel/colinear:
    # lines are not parallel/colinear and intersect
    if not are_parallel(e1, e2) and intersect(e1, e2):
        sq_distance = 0
    # lines don't intersect, aren't parallel
    else:
        d1 = sq_distance_point_to_segment(e1.nodes[0], e2)
        d2 = sq_distance_point_to_segment(e1.nodes[1], e2)
        d3 = sq_distance_point_to_segment(e2.nodes[0], e1)
        d4 = sq_distance_point_to_segment(e2.nodes[1], e1)
        sq_distance = min(d1, d2, d3, d4)

    return sq_distance


# vector math
def bisect_angle(a, b, c, epsilon=0.2, radius=1):
    """ finds point d such that bd bisects the lines ab and bc."""
    ax = a.x - b.x
    ay = a.y - b.y

    cx = c.x - b.x
    cy = c.y - b.y

    a1 = mg.MyNode(((ax, ay))/np.linalg.norm((ax, ay)))
    c1 = mg.MyNode(((cx, cy))/np.linalg.norm((cx, cy)))

    # if vectors are close to parallel, find vector that is perpendicular to ab
    # if they are not, then find the vector that bisects a and c
    if abs(np.cross(a1.loc, c1.loc)) < 0 + epsilon:
        # print("vectors {0}{1} and {1}{2} are close to //)".format(a,b,c)
        dx = -ay
        dy = ax
    else:
        dx = (a1.x + c1.x)/2
        dy = (a1.y + c1.y)/2

    # convert d values into a vector of length radius
    dscale = ((dx, dy)/np.linalg.norm((dx, dy)))*radius
    myd = mg.MyNode(dscale)

    # make d a node in space, not vector around b
    d = mg.MyNode((myd.x + b.x, myd.y + b.y))

    return d


def find_negative(d, b):
    """finds the vector -d when b is origen """
    negx = -1*(d.x - b.x) + b.x
    negy = -1*(d.y - b.y) + b.y
    dneg = mg.MyNode((negx, negy))
    return dneg


# clean up and probability functions
def WeightedPick(d):
    """picks an item out of the dictionary d, with probability proportional to
    the value of that item.  e.g. in {a:1, b:0.6, c:0.4} selects and returns
    "a" 5/10 times, "b" 3/10 times and "c" 2/10 times. """

    r = random.uniform(0, sum(d.values()))
    s = 0.0
    for k, w in d.items():
        s += w
        if r < s:
            return k
    return k


def mat_reorder(matrix, order):
    """sorts a square matrix so both rows and columns are
    ordered by order. """

    Drow = [matrix[i] for i in order]
    Dcol = [[r[i] for i in order] for r in Drow]

    return Dcol


def myRoll(mylist):
    """rolls a list, putting the last element into the first slot. """

    mylist.insert(0, mylist[-1])
    del mylist[-1]
    return(mylist)

######################
# DEALING WITH PATHS
#######################


def path_length(path):
    """finds the geometric path length for a path that consists of a list of
    MyNodes. """
    length = 0
    for i in range(1, len(path)):
        length += distance(path[i-1], path[i])
    return length

# def path_length_npy(path):
#    xy = np.array([n.x,n.y for n in path])
#    return np.linalg.norm(xy[1:] - xy[:-1],2,1).sum()


def shorten_path(ptup):
    """ all the paths found in my pathfinding algorithm start at the fake
    road side, and go towards the interior of the parcel.  This method drops
    nodes beginning at the fake road node, until the first and only the
    first node is on a road.  This gets rid of paths that travel along a
    curb before ending."""

#    roadtrue = [p.road for p in ptup]
#    if roadtrue[0] == roadtrue[-1]:
#        raise Exception("segment does not start off road and end on road
#        \n {} \n {}".format(ptup, roadtrue))
    while ptup[1].road is True:
        ptup.pop(0)
    return ptup


def segment_near_path(myG, segment, path, threshold):
    """returns True if the segment is within (geometric) distance threshold
    of all the segments contained in path is stored as a list of nodes that
    strung together make up a path.
    """
    # assert isinstance(segment, mg.MyEdge)

    pathlist = []

    for i in range(1, len(path)):
        pedge = myG.G[path[i-1]][path[i]]['myedge']
        pathlist.append(pedge)

    for p in pathlist:
        sq_distance = segment_distance_sq(p, segment)
        if sq_distance < threshold**2:
            return True

    return False


def _fake_edge(myA, centroid, mynode):
    newedge = mg.MyEdge((centroid, mynode))
    newedge.length = 0
    myA.add_edge(newedge)


def __add_fake_edges(myA, p, roads_only=False):
    if roads_only:
        [_fake_edge(myA, p.centroid, n) for n in p.nodes if n.road]
    else:
        [_fake_edge(myA, p.centroid, n) for n in p.nodes]


def shortest_path_setup(myA, p, roads_only=False):
    """ sets up graph to be ready to find the shortest path from a
    parcel to the road. if roads_only is True, only put fake edges for the
    interior parcel to nodes that are already connected to a road. """

    fake_interior = p.centroid

    __add_fake_edges(myA, p)

    fake_road_origin = mg.MyNode((305620, 8022470))

    for i in myA.road_nodes:
        if len(myA.G.neighbors(i)) > 2:
            _fake_edge(myA, fake_road_origin, i)
    return fake_interior, fake_road_origin


def shortest_path_p2p(myA, p1, p2):
    """finds the shortest path along fenclines from a given interior parcel
    p1 to another parcel p2"""

    __add_fake_edges(myA, p1, roads_only=True)
    __add_fake_edges(myA, p2, roads_only=True)

    path = nx.shortest_path(myA.G, p1.centroid, p2.centroid, "weight")
    length = nx.shortest_path_length(myA.G, p1.centroid, p2.centroid, "weight")

    myA.G.remove_node(p1.centroid)
    myA.G.remove_node(p2.centroid)

    return path[1:-1], length


def find_short_paths(myA, parcel, barriers=True):
    """ finds short paths from an interior parcel,
    returns them and stores in parcel.paths  """

    if barriers:
        barrier_edges = [e for e in myA.myedges() if e.barrier]
        if len(barrier_edges) > 0:
            myA.remove_myedges_from(barrier_edges)
        else:
            print "no barriers found. Did you expect them?"
        # myA.plot_roads(title = "myA no barriers")

    interior, road = shortest_path_setup(myA, parcel)

    shortest_path = nx.shortest_path(myA.G, interior, road, "weight")
    shortest_path_segments = len(shortest_path)
    shortest_path_distance = path_length(shortest_path[1:-1])
    all_simple = [shorten_path(p[1:-1]) for p in
                  nx.all_simple_paths(myA.G, road, interior,
                                      cutoff=shortest_path_segments + 2)]
    paths = {tuple(p): path_length(p) for p in all_simple
             if path_length(p) < shortest_path_distance*2}

    myA.G.remove_node(road)
    myA.G.remove_node(interior)
    if barriers:
        for e in barrier_edges:
            myA.add_edge(e)

    parcel.paths = paths

    return paths


def find_short_paths_all_parcels(myA, new_road1=None, new_road2=None,
                                 barriers=True, quiet=False):
    """ finds the short paths for all parcels, stored in parcel.paths"""
    all_paths = {}
    counter = 0

    for parcel in myA.interior_parcels:
        # if paths have already been defined for this parcel
        # (new_road should exist too)
        if parcel.paths:
            needs_update = False
            for pathitem in parcel.paths.items():
                    path = pathitem[0]
                    path_length = pathitem[1]
                    if new_road1 is not None:
                        if segment_near_path(myA, new_road1,
                                             path, path_length):
                            needs_update = True
                            break
                    if new_road2 is not None:
                        if segment_near_path(myA, new_road2,
                                             path, path_length):
                            needs_update = True
                            break
            if needs_update is True:
                paths = find_short_paths(myA, parcel,
                                         barriers=barriers)
                counter += 1
                all_paths.update(paths)
            elif needs_update is False:
                paths = parcel.paths
                all_paths.update(paths)
        # if paths have not been defined for this parcel
        else:
            paths = find_short_paths(myA, parcel, barriers=barriers)
            counter += 1
            all_paths.update(paths)
    if quiet is False:
        print ("Shortest paths found for {} parcels".format(counter))

    return all_paths


def build_path(myG, start, finish):
    ptup = nx.shortest_path(myG.G, start, finish, weight="weight")

    ptup = shorten_path(ptup)
    ptup.reverse()
    ptup = shorten_path(ptup)

    myedges = [myG.G[ptup[i-1]][ptup[i]]["myedge"]
               for i in range(1, len(ptup))]

    for e in myedges:
        myG.add_road_segment(e)

    return ptup, myedges

#############################################
#  PATH SELECTION AND CONSTRUCTION
#############################################


def choose_path_greedy(myG):

    """ chooses the path segment, currently based on a strictly
    greedy algorithm  """

    start_parcel = min(myG.path_len_dict, key=myG.path_len_dict.get)
    shortest_path = myG.paths_dict[start_parcel]
    new_road = myG.G[shortest_path[0]][shortest_path[1]]["myedge"]

    myG.new_road = new_road

    return myG.new_road


def choose_path_probablistic(myG, paths, alpha):

    """ chooses the path segment, choosing paths of shorter
    length more frequently  """

    inv_weight = {k: 1.0/(paths[k]**alpha) for k in paths}
    target_path = WeightedPick(inv_weight)
    new_road = myG.G[target_path[0]][target_path[1]]["myedge"]

    myG.new_road1 = new_road
    myG.new_road2 = myG.G[target_path[-2]][target_path[-1]]["myedge"]

    return myG.new_road1, myG.new_road2, target_path


def build_all_roads(myG, master=None, alpha=2, plot_intermediate=False,
                    wholepath=False, original_roads=None, plot_original=False,
                    bisect=False, plot_result=False, barriers=True,
                    quiet=False, vquiet=False):

    """builds roads using the probablistic greedy alg, until all
    interior parcels are connected, and returns the total length of
    road built. """

    if vquiet is True:
        quiet = True

    if plot_original:
        myG.plot_roads(original_roads, update=False,
                       parcel_labels=False, new_road_color="blue")

    added_road_length = 0
    plotnum = 0
    if plot_intermediate is True and master is None:
        master = myG.copy()

    myG.define_interior_parcels()
    nr1 = None
    nr2 = None

    if vquiet is False:
        print ("Begin w {} Interior Parcels".format(len(myG.interior_parcels)))

    while myG.interior_parcels:
        # find all potential segments
        all_paths = find_short_paths_all_parcels(myG, nr1, nr2,
                                                 barriers, quiet=quiet)

        # choose and build one
        nr1, nr2, new_path = choose_path_probablistic(myG, all_paths,
                                                      alpha)
        if wholepath is False:
            added_road_length += nr1.length
            myG.add_road_segment(nr1)

        if wholepath is True:
            for i in range(0, len(new_path) - 1):
                new_road = myG.G[new_path[i]][new_path[i+1]]["myedge"]
                added_road_length += new_road.length
                myG.add_road_segment(new_road)
        myG.define_interior_parcels()
        if plot_intermediate:
            myG.plot_roads(master, update=False)
            plt.savefig("Intermediate_Step"+str(plotnum)+".pdf", format='pdf')
            plotnum += 1

        remain = len(myG.interior_parcels)
        if quiet is False:
            print "{} interior parcels left".format(remain)
        if vquiet is False:
            if remain > 300 or remain in [50, 100, 150, 200, 225, 250, 275]:
                print "{} interior parcels".format(remain)

    # update the properties of nodes & edges to reflect new geometry.

    # once done getting all interior parcels connected, have option to bisect
    bisecting_roads = 0
    if bisect:
        start, finish = bisecting_path_endpoints(myG)
        ptup, myedges = build_path(myG, start, finish)
        bisecting_roads = path_length(ptup)

    if plot_result:
        myG.plot_roads(original_roads, update=False,
                       parcel_labels=False, new_road_color="blue")
        plt.title("All Parcels Connected," +
                  "\n New Road Len = {:.0f}".format(new_roads_i))

    myG.added_roads = added_road_length + bisecting_roads
    return added_road_length, bisecting_roads


############################
# connectivity optimization
############################

def __road_connections_through_culdesac(myG, threshold=5):
    """connects all nodes on a road that are within threshold = 5 meters of
    each other.  This means that paths can cross a culdesac instead of needing
    to go around. """

    etup_drop = []

    nlist = [i for i in myG.G.nodes() if i.road is True]
    for i, j in itertools.combinations(nlist, 2):
        if j in myG.G and i in myG.G:
            if distance(i, j) < threshold:
                newE = mg.MyEdge((i, j))
                newE.road = True
                myG.add_edge(newE)
                etup_drop.append((i, j))

    return etup_drop


def shortest_path_p2p_matrix(myG, full=False, travelcost=False):
    """option if full is false, removes all interior edges, so that travel
    occurs only along a road.  If full is true, keeps interior edges.  If
    travel cost is true, increases weight of non-road edges by a factor of ten.
    Base case is defaults of false and false."""

    copy = myG.copy()

    etup_drop = copy.find_interior_edges()
    if full is False:
        copy.G.remove_edges_from(etup_drop)
        # print("dropping {} edges".format(len(etup_drop)))

    __road_connections_through_culdesac(copy)

    path_mat = []
    path_len_mat = []

    edges = copy.myedges()

    if travelcost is True:
        for e in edges:
            if e.road is False:
                copy.G[e.nodes[0]][e.nodes[1]]['weight'] = e.length*10

    for p0 in copy.inner_facelist:
        path_vec = []
        path_len_vec = []
        __add_fake_edges(copy, p0)

        for p1 in copy.inner_facelist:
            if p0.centroid == p1.centroid:
                length = 0
                path = p0.centroid
            else:
                __add_fake_edges(copy, p1)
                try:
                    path = nx.shortest_path(copy.G, p0.centroid, p1.centroid,
                                            "weight")
                    length = path_length(path[1:-1])
                except:
                    path = []
                    length = np.nan
                copy.G.remove_node(p1.centroid)
            path_vec.append(path)
            path_len_vec.append(length)

        copy.G.remove_node(p0.centroid)
        path_mat.append(path_vec)
        path_len_mat.append(path_len_vec)

        n = len(path_len_mat)
        meantravel = (sum([sum(i) for i in path_len_mat])/(n*(n-1)))

    return path_len_mat, meantravel


def difference_roads_to_fences(myG, travelcost=False):
    fullpath_len, tc = shortest_path_p2p_matrix(myG, full=True)
    path_len, tc = shortest_path_p2p_matrix(myG, full=False)

    diff = [[path_len[j][i] - fullpath_len[j][i]
            for i in range(0, len(fullpath_len[j]))]
            for j in range(0, len(fullpath_len))]

    dmax = max([max(i) for i in diff])
    # print dmax

    for j in range(0, len(path_len)):
        for i in range(0, len(path_len[j])):
            if np.isnan(path_len[j][i]):
                diff[j][i] = dmax + 150
                #            diff[j][i] = np.nan

    n = len(path_len)
    meantravel = sum([sum(i) for i in path_len])/(n*(n-1))

    return diff, fullpath_len, path_len, meantravel


def bisecting_path_endpoints(myG):
    roads_only = myG.copy()
    etup_drop = roads_only.find_interior_edges()
    roads_only.G.remove_edges_from(etup_drop)
    __road_connections_through_culdesac(roads_only)
    # nodes_drop = [n for n in roads_only.G.nodes() if not n.road]
    # roads_only.G.remove_nodes_from(nodes_drop)

    distdict = {}

    for i, j in itertools.combinations(myG.road_nodes, 2):
        geodist_sq = distance_squared(i, j)
        onroad_dist = nx.shortest_path_length(roads_only.G, i, j,
                                              weight='weight')
        dist_diff = onroad_dist**2/geodist_sq
        distdict[(i, j)] = dist_diff
    (i, j) = max(distdict.iteritems(), key=operator.itemgetter(1))[0]

    return i, j

################
# GRAPH INSTANTIATION
###################


def graphFromMyEdges(elist, name=None):
    myG = mg.MyGraph(name=name)
    for e in elist:
        myG.add_edge(e)
    return myG


def graphFromMyFaces(flist, name=None):
    myG = mg.MyGraph(name=name)
    for f in flist:
        for e in f.edges:
            myG.add_edge(e)
    return myG


def graphFromShapes(shapes, name, rezero=np.array([0, 0])):
    nodedict = dict()
    plist = []
    for s in shapes:
        nodes = []
        for k in s.points:
            k = k - rezero
            myN = mg.MyNode(k)
            if myN not in nodedict:
                nodes.append(myN)
                nodedict[myN] = myN
            else:
                nodes.append(nodedict[myN])
            edges = [(nodes[i], nodes[i+1]) for i in range(0, len(nodes)-1)]
            plist.append(mg.MyFace(edges))

    myG = mg.MyGraph(name=name)

    for p in plist:
        for e in p.edges:
            myG.add_edge(mg.MyEdge(e.nodes))

    print("data loaded")

    return myG


def build_barriers(myG, edgelist):
    # assert isinstance(edgelist[0], mg.MyEdge), "{} is not and edge".
    # format(edgelist[0])
    for e in edgelist:
        if e in myG.myedges():
            myG.remove_road_segment(e)
            e.barrier = True

####################
# PLOTTING FUNCTIONS
####################


def plot_cluster_mat(clustering_data, plotting_data, title, dmax,
                     plot_dendro=True):
    """from http://nbviewer.ipython.org/github/OxanaSachenkova/
    hclust-python/blob/master/hclust.ipynb  First input matrix is used to
    define clustering order, second is the data that is plotted."""

    fig = plt.figure(figsize=(8, 8))
    # x ywidth height

    ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.6])
    Y = linkage(clustering_data, method='single')
    Z1 = dendrogram(Y, orientation='right')  # adding/removing the axes
    ax1.set_xticks([])
    # ax1.set_yticks([])

# Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3, 0.75, 0.6, 0.1])
    Z2 = dendrogram(Y)
    # ax2.set_xticks([])
    ax2.set_yticks([])

    # set up custom color map
    c = mcolors.ColorConverter().to_rgb
    seq = [c('navy'), c('mediumblue'), .1, c('mediumblue'),
           c('darkcyan'), .2, c('darkcyan'), c('darkgreen'), .3,
           c('darkgreen'), c('lawngreen'), .4, c('lawngreen'), c('yellow'), .5,
           c('yellow'), c('orange'), .7, c('orange'), c('red')]
    custommap = make_colormap(seq)

    # Compute and plot the heatmap
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    if not plot_dendro:
        fig = plt.figure(figsize=(8, 8))
        axmatrix = fig.add_axes([0.05, 0.1, 0.85, 0.8])

    idx1 = Z1['leaves']
    D = mat_reorder(plotting_data, idx1)
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=custommap,
                          vmin=0, vmax=dmax)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    h = 0.6
    if not plot_dendro:
        h = 0.8
    axcolor = fig.add_axes([0.91, 0.1, 0.02, h])
    plt.colorbar(im, cax=axcolor)
    ax2.set_title(title)
    if not plot_dendro:
        axmatrix.set_title(title)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def plotly_notebook(traces, filename=None, title=None):
    if filename is None:
        filename = "plotly_graph"
    fig = Figure(data=Data(traces),
                 layout=Layout(title=title, plot_bgcolor="rgb(217, 217, 217)",
                               showlegend=True,
                               xaxis=XAxis(showgrid=False, zeroline=False,
                                           showticklabels=False),
                               yaxis=YAxis(showgrid=False, zeroline=False,
                                           showticklabels=False)))
    py.iplot(fig, filename=filename)


######################
#  IMPORT & Running FUNCTIONS #
#####################


def import_and_setup(component, filename, threshold=1,
                     rezero=np.array([0, 0]), connected=False, name=""):
    plt.close('all')

    # check that rezero is an array of len(2)
    # check that threshold is a float

    sf = shapefile.Reader(filename)
    myG = graphFromShapes(sf.shapes(), name, rezero)

    myG = myG.clean_up_geometry(threshold, connected)

    if connected is True:
        return myG
    else:
        return myG.connected_components()[component]


####################
# Testing functions
###################


def test_edges_equality():
    """checks that myGraph points to myEdges correctly   """
    testG = testGraph()
    testG.trace_faces()
    outerE = list(testG.outerface.edges)[0]
    return outerE is testG.G[outerE.nodes[0]][outerE.nodes[1]]['myedge']


def test_weak_duals():
    """ plots the weak duals based on testGraph"""
    S0 = testGraph()
    S1 = S0.weak_dual()
    S2 = S1.weak_dual()
    S3 = S2.weak_dual()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    S0.plot(ax=ax, node_color='b', edge_color='k', node_size=300)
    S1.plot(ax=ax, node_color='g', edge_color='b', node_size=200)
    S2.plot(ax=ax, node_color='r', edge_color='g', node_size=100)
    S3.plot(ax=ax, node_color='c', edge_color='r', node_size=50)
    ax.legend()
    ax.set_title("Test Graph")
    plt.show()


def test_nodes(n1, n2):
    """ returns true if two nodes are evaluated as the same"""
    eq_num = len(set(n1).intersection(set(n2)))
    is_num = len(set([id(n) for n in n1])
                 .intersection(set([id(n) for n in n2])))
    print("is eq? ", eq_num, "is is? ", is_num)


def testGraph():
    n = {}
    n[1] = mg.MyNode((0, 0))
    n[2] = mg.MyNode((0, 1))
    n[3] = mg.MyNode((0, 2))
    n[4] = mg.MyNode((0, 3))
    n[5] = mg.MyNode((1, 2))
    n[6] = mg.MyNode((1, 3))
    n[7] = mg.MyNode((0, 4))
    n[8] = mg.MyNode((-1, 4))
    n[9] = mg.MyNode((-1, 3))
    n[10] = mg.MyNode((-1, 2))
    n[11] = mg.MyNode((1, 4))
    n[12] = mg.MyNode((-2, 3))

    lat = mg.MyGraph(name="S0")
    lat.add_edge(mg.MyEdge((n[1], n[2])))
    lat.add_edge(mg.MyEdge((n[2], n[3])))
    lat.add_edge(mg.MyEdge((n[2], n[5])))
    lat.add_edge(mg.MyEdge((n[3], n[4])))
    lat.add_edge(mg.MyEdge((n[3], n[5])))
    lat.add_edge(mg.MyEdge((n[3], n[9])))
    lat.add_edge(mg.MyEdge((n[4], n[5])))
    lat.add_edge(mg.MyEdge((n[4], n[6])))
    lat.add_edge(mg.MyEdge((n[4], n[7])))
    lat.add_edge(mg.MyEdge((n[4], n[8])))
    lat.add_edge(mg.MyEdge((n[4], n[9])))
    lat.add_edge(mg.MyEdge((n[5], n[6])))
    lat.add_edge(mg.MyEdge((n[6], n[7])))
    lat.add_edge(mg.MyEdge((n[7], n[8])))
    lat.add_edge(mg.MyEdge((n[8], n[9])))
    lat.add_edge(mg.MyEdge((n[9], n[10])))
    lat.add_edge(mg.MyEdge((n[3], n[10])))
    lat.add_edge(mg.MyEdge((n[2], n[10])))
    lat.add_edge(mg.MyEdge((n[7], n[11])))
    lat.add_edge(mg.MyEdge((n[6], n[11])))
    lat.add_edge(mg.MyEdge((n[10], n[12])))
    lat.add_edge(mg.MyEdge((n[8], n[12])))

    return lat


def testGraphLattice():
    n = {}
    n[1] = mg.MyNode((0, 0))
    n[2] = mg.MyNode((0, 10))
    n[3] = mg.MyNode((0, 20))
    n[4] = mg.MyNode((0, 30))
    n[5] = mg.MyNode((0, 40))
    n[6] = mg.MyNode((10, 0))
    n[7] = mg.MyNode((10, 10))
    n[8] = mg.MyNode((10, 20))
    n[9] = mg.MyNode((10, 30))
    n[10] = mg.MyNode((10, 40))
    n[11] = mg.MyNode((20, 0))
    n[12] = mg.MyNode((20, 10))
    n[13] = mg.MyNode((20, 20))
    n[14] = mg.MyNode((20, 30))
    n[15] = mg.MyNode((20, 40))
    n[16] = mg.MyNode((30, 0))
    n[17] = mg.MyNode((30, 10))
    n[18] = mg.MyNode((30, 20))
    n[19] = mg.MyNode((30, 30))
    n[20] = mg.MyNode((30, 40))
    n[21] = mg.MyNode((40, 0))
    n[22] = mg.MyNode((40, 10))
    n[23] = mg.MyNode((40, 20))
    n[24] = mg.MyNode((40, 30))
    n[25] = mg.MyNode((40, 40))

    lat = mg.MyGraph(name="S0")
    lat.add_edge(mg.MyEdge((n[1], n[2])))
    lat.add_edge(mg.MyEdge((n[1], n[6])))
    lat.add_edge(mg.MyEdge((n[2], n[3])))
    lat.add_edge(mg.MyEdge((n[2], n[7])))
    lat.add_edge(mg.MyEdge((n[3], n[4])))
    lat.add_edge(mg.MyEdge((n[3], n[8])))
    lat.add_edge(mg.MyEdge((n[4], n[5])))
    lat.add_edge(mg.MyEdge((n[4], n[9])))
    lat.add_edge(mg.MyEdge((n[5], n[10])))
    lat.add_edge(mg.MyEdge((n[6], n[7])))
    lat.add_edge(mg.MyEdge((n[6], n[11])))
    lat.add_edge(mg.MyEdge((n[7], n[8])))
    lat.add_edge(mg.MyEdge((n[7], n[12])))
    lat.add_edge(mg.MyEdge((n[8], n[9])))
    lat.add_edge(mg.MyEdge((n[8], n[13])))
    lat.add_edge(mg.MyEdge((n[9], n[10])))
    lat.add_edge(mg.MyEdge((n[9], n[14])))
    lat.add_edge(mg.MyEdge((n[10], n[15])))
    lat.add_edge(mg.MyEdge((n[11], n[12])))
    lat.add_edge(mg.MyEdge((n[11], n[16])))
    lat.add_edge(mg.MyEdge((n[12], n[13])))
    lat.add_edge(mg.MyEdge((n[12], n[17])))
    lat.add_edge(mg.MyEdge((n[13], n[14])))
    lat.add_edge(mg.MyEdge((n[13], n[18])))
    lat.add_edge(mg.MyEdge((n[14], n[15])))
    lat.add_edge(mg.MyEdge((n[14], n[19])))
    lat.add_edge(mg.MyEdge((n[15], n[20])))
    lat.add_edge(mg.MyEdge((n[15], n[20])))
    lat.add_edge(mg.MyEdge((n[16], n[17])))
    lat.add_edge(mg.MyEdge((n[16], n[21])))
    lat.add_edge(mg.MyEdge((n[17], n[18])))
    lat.add_edge(mg.MyEdge((n[17], n[22])))
    lat.add_edge(mg.MyEdge((n[18], n[19])))
    lat.add_edge(mg.MyEdge((n[18], n[23])))
    lat.add_edge(mg.MyEdge((n[19], n[20])))
    lat.add_edge(mg.MyEdge((n[19], n[24])))
    lat.add_edge(mg.MyEdge((n[20], n[25])))
    lat.add_edge(mg.MyEdge((n[21], n[22])))
    lat.add_edge(mg.MyEdge((n[22], n[23])))
    lat.add_edge(mg.MyEdge((n[23], n[24])))
    lat.add_edge(mg.MyEdge((n[24], n[25])))

    return lat, n


def testGraphEquality():
    n = {}
    n[1] = mg.MyNode((0, 0))
    n[2] = mg.MyNode((0, 1))
    n[3] = mg.MyNode((1, 1))
    n[4] = mg.MyNode((1, 0))
    n[5] = mg.MyNode((0, 0))  # actually equal
    n[6] = mg.MyNode((0.0001, 0.0001))  # within rounding
    n[7] = mg.MyNode((0.1, 0.1))  # within threshold
    n[8] = mg.MyNode((0.3, 0.3))  # actually different

    G = mg.MyGraph(name="S0")
    G.add_edge(mg.MyEdge((n[1], n[2])))
    G.add_edge(mg.MyEdge((n[2], n[3])))
    G.add_edge(mg.MyEdge((n[3], n[4])))
    G.add_edge(mg.MyEdge((n[4], n[5])))

    return G, n


def __centroid_test():
    n = {}
    n[1] = mg.MyNode((0, 0))
    n[2] = mg.MyNode((0, 1))
    n[3] = mg.MyNode((1, 1))
    n[4] = mg.MyNode((1, 0))
    n[5] = mg.MyNode((0.55, 0))
    n[6] = mg.MyNode((0.5, 0.9))
    n[7] = mg.MyNode((0.45, 0))
    n[8] = mg.MyNode((0.4, 0))
    n[9] = mg.MyNode((0.35, 0))
    n[10] = mg.MyNode((0.3, 0))
    n[11] = mg.MyNode((0.25, 0))
    nodeorder = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1]
    nodetups = [(n[nodeorder[i]], n[nodeorder[i+1]])
                for i in range(0, len(nodeorder)-1)]
    edgelist = [mg.MyEdge(i) for i in nodetups]

    f1 = mg.MyFace(nodetups)
    S0 = graphFromMyFaces([f1])

    S0.define_roads()
    S0.define_interior_parcels()

    S0.plot_roads(parcel_labels=True)

    return S0, f1, n, edgelist


def testmat():
    testmat = []
    dim = 4
    for i in range(0, dim):
        k = []
        for j in range(0, dim):
            k.append((i-j)*(i-j))
        testmat.append(k)
    return testmat


def build_lattice_barrier(myG):
    edgesub = [e for e in myG.myedges()
               if e.nodes[0].y == 0 and e.nodes[1].y == 0]
    # barrieredges = [e for e in edgesub if e.nodes[1].y == 0]

    for e in edgesub:
        myG.remove_road_segment(e)
        e.barrier = True

    myG.define_interior_parcels()
    return myG, edgesub


if __name__ == "__main__":
    S0, n = testGraphLattice()
    S0.define_roads()
    S0.define_interior_parcels()
    # S0, barrier_edges = build_lattice_barrier(S0)
    barGraph = graphFromMyEdges(barrier_edges)
    master = S0.copy()

    # S0.plot_roads(master, update=False, new_plot=True)

    new_roads_i = build_all_roads(S0, S0, alpha=2, wholepath=True,
                                  barriers=False)

    S0.plot_roads(master, update=False)
    # barGraph.plot(node_size=25, node_color='green', width=3,
    #              edge_color='green')

    S1 = S0.weak_dual()
    S1.plot()
