import numpy as np
import networkx as nx
import itertools
import math
from matplotlib import pyplot as plt
import my_graph_helpers as mgh
from lazy_property import lazy_property


class MyNode(object):
    """ rounds float nodes to (2!) decimal places, defines equality """
    def __init__(self, locarray, name=None):
        if len(locarray) != 2:
            print "error"
        x = locarray[0]  # - 1605000
        y = locarray[1]  # - 6429200
        self.x = np.round(float(x), 2)
        self.y = np.round(float(y), 2)
        self.loc = (self.x, self.y)
        self.road = False
        self.on_interior = False
        self.name = name

    def __repr__(self):
        if self.name:
            return self.name
        else:
            return "(%.2f,%.2f)" % (self.x, self.y)  # + str(id(self))

    def __eq__(self, other):
        return self.loc == other.loc

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.loc < other.loc

    def __hash__(self):
        return hash(self.loc)


class MyEdge(object):
    """ keeps the properties of the edges in a parcel."""

    def __init__(self, nodes):
        self.nodes = tuple(nodes)
        self.parcel1 = None
        self.parcel2 = None
        self.road = False
        self.barrier = False

    @lazy_property
    def length(self):
        return mgh.distance(self.nodes[0], self.nodes[1])

    @lazy_property
    def rads(self):
        return math.atan((self.nodes[0].y - self.nodes[1].y) /
                         (self.nodes[0].x - self.nodes[1].x))

    def __repr__(self):
        return "MyEdge with nodes {} {}".format(self.nodes[0], self.nodes[1])

    def __eq__(self, other):
        return ((self.nodes[0] == other.nodes[0] and
                 self.nodes[1] == other.nodes[1]) or
                (self.nodes[0] == other.nodes[1] and
                 self.nodes[1] == other.nodes[0]))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.nodes)


class MyFace(object):
    """class defines a face (with name and list of edges & nodes)
       from a list of edges in the face"""

    def __init__(self, list_of_edges):
        # make a list of all the nodes in the face

        isMyEdge = False
        if len(list_of_edges) > 0:
            isMyEdge = type(list_of_edges[0]) != tuple

        if isMyEdge:
            node_set = set(n for edge in list_of_edges for n in edge.nodes)
        else:
            node_set = set(n for edge in list_of_edges for n in edge)

        self.nodes = sorted(list(node_set))
        alpha_nodes = map(str, self.nodes)
        self.name = ".".join(alpha_nodes)
        self.paths = None
        self.on_road = False

        # the position of the face is the centroid of the nodes that
        # compose the face

        if isMyEdge:
            self.edges = set(list_of_edges)
        else:
            self.edges = set(MyEdge(e) for e in list_of_edges)

    @lazy_property
    def area(self):
        return 0.5*abs(sum(e.nodes[0].x*e.nodes[1].y -
                       e.nodes[1].x*e.nodes[0].y for e in self.edges))

    @lazy_property
    def centroid(self):
        """finds the centroid of a myface """

        a = 0.5*(sum(e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y
                 for e in self.edges))
        if abs(a) < 0.01:
            cx = np.mean([n.x for n in self.nodes])
            cy = np.mean([n.y for n in self.nodes])
        else:
            cx = (1/(6*a))*sum([(e.nodes[0].x + e.nodes[1].x) *
                               (e.nodes[0].x*e.nodes[1].y -
                               e.nodes[1].x*e.nodes[0].y)
                               for e in self.edges])
            cy = (1/(6*a))*sum([(e.nodes[0].y + e.nodes[1].y) *
                               (e.nodes[0].x*e.nodes[1].y -
                               e.nodes[1].x*e.nodes[0].y)
                               for e in self.edges])

        return MyNode((cx, cy))


    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return "Face with centroid at (%.2f,%.2f)" % (self.centroid.x,
                                                      self.centroid.y)


class MyParcel(MyFace):
    def __init__(self, nodes, parcel_name="P0"):
        self.parcel_name = parcel_name
        edges = [(nodes[i], nodes[i+1]) for i in range(0, len(nodes)-1)]
        MyFace.__init__(self, edges)


class MyGraph(object):
    def __init__(self, G=None, name="S0"):

        """ MyGraph is a regular networkx graph where nodes are stored
        as MyNodes and edges have the attribute myedge = MyEdge.

        The extra function weak_dual() finds the weak dual
        (http://en.wikipedia.org/wiki/Dual_graph#Weak_dual) of the
        graph based on the locations of each node.  Each node in the
        dual graph corresponds to a face in G, the position of each
        node in the dual is caluclated as the mean of the nodes
        composing the corresponding face in G."""

        self.name = name
        self.cleaned = False
        self.roads_update = True

        if G is None:
            self.G = nx.Graph()
        else:
            self.G = G

    def __repr__(self):
        return "Graph (%s) with %d nodes" % (self.name,
                                             self.G.number_of_nodes())

    def add_node(self, n):
        self.G.add_node(n)

    def add_edge(self, e, weight=None):
        if weight is None:
            w = e.length
        else:
            w = weight
        self.G.add_edge(e.nodes[0], e.nodes[1], myedge=e, weight=w)

    def location_dict(self):
        return {n: n.loc for n in self.G.nodes_iter()}

    def connected_components(self):
        return [MyGraph(g, self.name+" pt %d" % i) for i, g
                in enumerate(nx.connected_component_subgraphs(self.G))]

    def myedges(self):
        return [self.G[e[0]][e[1]]["myedge"] for e in self.G.edges()]

    def remove_myedges_from(self, myedges):
        myedge_tups = [(e.nodes[0], e.nodes[1]) for e in myedges]
        self.G.remove_edges_from(myedge_tups)

    def copy(self):
        nx_copy = self.G.copy()
        copy = MyGraph(nx_copy)
        copy.name = "copy_"+self.name
        try:
            copy.inner_facelist = self.inner_facelist
            copy.outerface = self.outerface
        except:
            pass
        try:
            copy.interior_parcels = self.interior_parcels
            copy.interior_nodes = self.interior_nodes
        except:
            pass
        try:
            copy.road_nodes = self.road_nodes
        except:
            pass
        return copy

############################
# GEOMETRY CLEAN UP FUNCTIONS
############################

    def __combine_near_nodes(self, threshold):

        """takes a connected component MyGraph, finds all nodes that are
        within a threshold distance of each other, drops one and keeps the
        other, and reconnects all the nodes that were connected to the first
        node to the second node.  """

        nlist = self.G.nodes()
        for i, j in itertools.combinations(nlist, 2):
            if j in self.G and i in self.G:
                if mgh.distance_squared(i, j) < threshold**2:
                    drop = j
                    keep = i
                    neighbors_drop = self.G.neighbors(drop)
                    neighbors_keep = self.G.neighbors(keep)
                    edges_to_add = (set(neighbors_drop) -
                                    set([keep]) -
                                    set(neighbors_keep))
                    self.G.remove_node(drop)
                    for k in edges_to_add:
                        newedge = MyEdge((keep, k))
                        self.add_edge(newedge)

    def __find_bad_edges(self, threshold):

        """ finds nodes that are within the threshold distance of an edge
        that does not contain it. Does not pair node V to edge UV.

        Returns a dict with edges as keys, and the node that is too close as
        the value.  This might cause trouble, if there are nodes that just
        should be collapsed together, rather than the edge being split in
        order to get that node connected. """

        node_list = self.G.nodes()
        edge_tup_list = self.G.edges(data=True)
        edge_list = [e[2]['myedge'] for e in edge_tup_list]
        bad_edge_dict = {}
        for i in node_list:
            for e in edge_list:
                if i != e.nodes[0] and i != e.nodes[1]:
                    # print "{} IS NOT on {}".format(i,e)
                    node_dist_sq = mgh.sq_distance_point_to_segment(i, e)
                    if node_dist_sq < threshold**2:
                        # print "{} is too close to {}".format(i, e)
                        if e in bad_edge_dict:
                            bad_edge_dict[e].append(i)
                        else:
                            bad_edge_dict[e] = list([i])
        self.bad_edge_dict = bad_edge_dict
        return

    def __remove_bad_edges(self, bad_edge_dict):
        """ From the dict of bad edges:  call edge (dict key) UV and the node
        (dict value) J. Then, drop edge UV and ensure that there is an edge
        UJ and JV.
        """

        dropped_edges = 0
        for edge, node_list in bad_edge_dict.iteritems():
            # print "dropping edge {}".format((edge.nodes[0],edge.nodes[1]))
            self.G.remove_edge(edge.nodes[0], edge.nodes[1])
            dropped_edges = dropped_edges + 1
            if len(node_list) == 1:
                for j in [0, 1]:
                    if not self.G.has_edge(node_list[0], edge.nodes[j]):
                        self.add_edge(MyEdge((node_list[0], edge.nodes[j])))
            else:
                node_list.sort(key=lambda
                               node: mgh.distance(node, edge.nodes[0]))
                if not self.G.has_edge(node_list[0], edge.nodes[0]):
                        self.add_edge(MyEdge((node_list[0], edge.nodes[0])))
                for i in range(1, len(node_list)):
                    if not self.G.has_edge(node_list[i], node_list[i-1]):
                        self.add_edge(MyEdge((node_list[i], node_list[i-1])))
                if not self.G.has_edge(node_list[-1], edge.nodes[1]):
                        self.add_edge(MyEdge((node_list[-1], edge.nodes[1])))

        return dropped_edges

    def clean_up_geometry(self, threshold, connected=True):

        """ function cleans up geometry, and returns a _copy_  of the graph,
        cleaned up nicely. Does not change original graph.
        """

        Gs = []
        if connected:
            for i in self.connected_components():
                i.G.remove_edges_from(i.G.selfloop_edges())
                i.__combine_near_nodes(threshold)
                i.__find_bad_edges(threshold)
                i.__remove_bad_edges(i.bad_edge_dict)
                Gs.append(i.G)
        else:
            i = self.copy()
            i.G.remove_edges_from(i.G.selfloop_edges())
            i.__combine_near_nodes(threshold)
            i.__find_bad_edges(threshold)
            i.__remove_bad_edges(i.bad_edge_dict)
            Gs.append(i.G)

        nxG = nx.compose_all(Gs)
        newG = MyGraph(nxG, name=self.name)
        newG.cleaned = True

        return newG

    def clean_up_geometry_single_CC(self, threshold):

        """ function cleans up geometry, and returns a _copy_  of the graph,
        cleaned up nicely. Does not change original graph.
        """

        Gs = self.copy()

        Gs.G.remove_edges_from(self.G.selfloop_edges())
        Gs.__combine_near_nodes(threshold)
        Gs.__find_bad_edges(threshold)
        Gs.__remove_bad_edges(Gs.bad_edge_dict)

        Gs.name = self.name

        Gs.cleaned = True

        return Gs

##########################################
#    WEAK DUAL CALCULATION FUNCTIONS
########################################

    def get_embedding(self):
        emb = {}
        for i in self.G.nodes():
            neighbors = self.G.neighbors(i)

            def angle(b):
                dx = b.x - i.x
                dy = b.y - i.y
                return np.arctan2(dx, dy)

            reorder_neighbors = sorted(neighbors, key=angle)
            emb[i] = reorder_neighbors
        return emb

    def trace_faces(self):
        """Algorithm from SAGE"""
        if len(self.G.nodes()) < 2:
            return []

        # grab the embedding
        comb_emb = self.get_embedding()

        # Establish set of possible edges
        edgeset = set()
        for edge in self.G.edges():
            edgeset = edgeset.union(set([(edge[0], edge[1]),
                                         (edge[1], edge[0])]))

        # Storage for face paths
        faces = []

        # Trace faces
        face = [edgeset.pop()]
        while (len(edgeset) > 0):
            neighbors = comb_emb[face[-1][-1]]
            next_node = neighbors[(neighbors.index(face[-1][-2])+1) %
                                  (len(neighbors))]
            edge_tup = (face[-1][-1], next_node)
            if edge_tup == face[0]:
                # myface = [self.G[e[1]][e[0]]["myedge"] for e in face]
                # faces.append(myface)
                faces.append(face)
                face = [edgeset.pop()]
            else:
                face.append(edge_tup)
                edgeset.remove(edge_tup)

        if len(face) > 0:
            # myface = [self.G[e[0]][e[1]]["myedge"] for e in face]
            # faces.append(myface)
            faces.append(face)

        # remove the outer "sphere" face
        facelist = sorted(faces, key=len)
        self.outerface = MyFace(facelist[-1])
        self.outerface.edges = [self.G[e[1]][e[0]]["myedge"]
                                for e in facelist[-1]]
        # self.outerface = facelist[-1]
        # self.inner_facelist = [MyFace(face) for face in facelist[:-1]]
        self.inner_facelist = []
        for face in facelist[:-1]:
            iface = MyFace(face)
            iface.edges = [self.G[e[1]][e[0]]["myedge"] for e in face]
            self.inner_facelist.append(iface)

    def weak_dual(self):
        """This function will create a networkx graph of the weak dual
        of a planar graph G with locations for each node.Each node in
        the dual graph corresponds to a face in G. The position of each
        node in the dual is caluclated as the mean of the nodes composing
        the corresponding face in G."""

        try:
            assert len(list(nx.connected_component_subgraphs(self.G))) <= 1
        except AssertionError:
            raise RuntimeError("weak_dual() can only be called on" +
                               " graphs which are fully connected.")

        # name the dual
        if len(self.name) == 0:
            dual_name = ""
        else:
            lname = list(self.name)
            nums = []
            while True:
                try:
                    nums.append(int(lname[-1]))
                except ValueError:
                    break
                else:
                    lname.pop()

            if len(nums) > 0:
                my_num = int(''.join(map(str, nums)))
            else:
                my_num = -1
            my_str = ''.join(lname)
            dual_name = my_str+str(my_num+1)

        # check for empty graph
        if self.G.number_of_nodes() < 2:
            return MyGraph(name=dual_name)

        # get a list of all faces
        self.trace_faces()

        # make a new graph, with faces from G as nodes and edges
        # if the faces share an edge
        dual = MyGraph(name=dual_name)
        if len(self.inner_facelist) == 1:
            dual.add_node(self.inner_facelist[0].centroid)
        else:
            combos = list(itertools.combinations(self.inner_facelist, 2))
            for c in combos:
                if len(set(c[0].edges).intersection(c[1].edges)) > 0:
                    dual.add_edge(MyEdge((c[0].centroid, c[1].centroid)))
        return dual

    def stacked_duals(self):
        def level_up(Slist):
            Sns = [g.weak_dual().connected_components() for g in Slist]
            Sn = [cc for duals in Sns for cc in duals]
            return Sn

        stacks = []
        stacks.append([self])
        while len(stacks) < 10:
            slist = level_up(stacks[-1])
            if len(slist) == 0:
                break
            stacks.append(slist)
        return stacks


#############################################
#  DEFINING ROADS AND INTERIOR PARCELS
#############################################

    def define_roads(self):
        """ finds which edges and nodes in the connected component are on
        the roads, and updates thier properties (node.road, edge.road) """

        road_nodes = []
        # check for empty graph
        if self.G.number_of_nodes() < 2:
            return []

        self.trace_faces()
        of = self.outerface

        for e in of.edges:
            e.road = True
        for n in of.nodes:
            n.road = True
            road_nodes.append(n)

        self.roads_update = True
        self.road_nodes = road_nodes
        # print "define roads called"

    def define_interior_parcels(self):

        """defines what parcels are on the interior based on
           whether their nodes are on roads   """

        if self.G.number_of_nodes() < 2:
            return []

        interior_nodes = []
        interior_parcels = []

        for f in self.inner_facelist:
            if len(set(f.nodes).intersection(set(self.road_nodes))) == 0:
                f.on_road = False
                interior_parcels.append(f)
                for n in f.nodes:
                    n.on_interior = True
                    interior_nodes.append(n)
            else:
                f.on_road = True
                for n in f.nodes:
                    n.on_interior = False
        self.interior_parcels = interior_parcels
        self.interior_nodes = interior_nodes
        # print "define interior parcels called"

    def find_interior_edges(self):
        """ finds and returns the pairs of nodes (not the myEdge) for all edges that
        are not on roads."""

        interior_etup = []

        for etup in self.G.edges():
            if not self.G[etup[0]][etup[1]]["myedge"].road:
                interior_etup.append(etup)

        return interior_etup

    def add_road_segment(self, edge):
        """ Updates properties of graph to make some edge a road. """
        edge.road = True
        for n in edge.nodes:
            n.road = True
            self.road_nodes.append(n)

        self.roads_update = False
        # self.define_interior_parcels()

    def remove_road_segment(self, edge):
        """ Updates properties of graph to remove a road. """
        assert isinstance(edge, MyEdge)
        edge.road = False
        for n in edge.nodes:
            onroad = False
            for neighbor in self.G[n]:
                neighboredge = self.G[n][neighbor]['myedge']
                if neighboredge.road:
                    onroad = True

            n.road = onroad
            if not n.road:
                if n in self.road_nodes:
                    self.road_nodes.remove(n)

        self.define_interior_parcels()
        return


#############################################
#   GEOMETRY AROUND BUILDING A GIVEN ROAD SEGMENT - c/(sh?)ould be deleted.
#############################################

    def __find_nodes_curbs(self, edge):
        """ finds curbs and nodes for a given edge that ends on a road.
        """

        if edge.nodes[0].road == edge.nodes[1].road:
            raise Exception("{} does not end on a curb".format(edge))

        [b] = [n for n in edge.nodes if n.road]
        [a] = [n for n in edge.nodes if not n.road]

        b_neighbors = self.G.neighbors(b)

        curb_nodes = [n for n in b_neighbors if self.G[b][n]["myedge"].road]

        if len(curb_nodes) != 2:
            raise Exception("Trouble!  " +
                            "Something is weird about the road geometery.")

        [c1, c2] = curb_nodes

        return a, b, c1, c2

    def __find_d_connections(self, a, b, c1, c2, d1, d2):
        """ nodes d1 and d2 are added to graph, and figures
            out how a, c1 and c2 are connected  """

        for n in [d1, d2]:
            self.add_edge(MyEdge((n, b)))

        emb = self.get_embedding()

        Bfilter = [n for n in emb[b] if n in [a, c1, c2, d1, d2]]

        if len(Bfilter) != 5:
            raise Exception("Bfilter is not set up correctly. \n {}"
                            .format(Bfilter))

        Aindex = Bfilter.index(a)

        while Aindex != 2:
            mgh.myRoll(Bfilter)
            Aindex = Bfilter.index(a)

        newedges = []
        newedges.append(MyEdge((a, d1)))
        newedges.append(MyEdge((a, d2)))
        newedges.append(MyEdge((Bfilter[0], Bfilter[1])))
        newedges.append(MyEdge((Bfilter[3], Bfilter[4])))

        return newedges

    def __find_e_connections(self, a, b, c1, c2, d1, d2):
        """ of nodes connected to b that are not existing curbs (c1 and c2)
        and a, the endpoint of the new road segment, figures out how to
        connect each one to d1 or d2 (the new curbs).   """

        emb = self.get_embedding()
        Bfilter = [n for n in emb[b] if n not in [d1, d2]]

        # if c1 and c2 are the first two elements, roll so they are the
        # first and last
        if ((Bfilter[0] == c1 or Bfilter[0] == c2) and
                (Bfilter[1] == c1 or Bfilter[1] == c2)):
            mgh.myRoll(Bfilter)

        # roll until c1 or c2 is first.  the other should then be the
        # last element in the list.
        while Bfilter[0] != c1 and Bfilter[0] != c2:
            mgh.myRoll(Bfilter)

        # check that after rolling, c1 and c2 are first and last elements
        if Bfilter[-1] != c1 and Bfilter[-1] != c2:
            raise Exception("There is an edge in my road." +
                            "Something is wrong with the geometry.")

        # d1 connected to c1 or c2?
        if c1 in self.G[d1]:
            c1_to_d1 = True
        else:
            c1_to_d1 = False

        if Bfilter[0] == c1:
            if c1_to_d1:
                dorder = [d1, d2]
            else:
                dorder = [d2, d1]
        elif Bfilter[0] == c2:
            if c1_to_d1:
                dorder = [d2, d1]
            else:
                dorder = [d1, d2]
        else:
            raise Exception("Bfilter is set up wrong")

        Aindex = Bfilter.index(a)

        newedges1 = [MyEdge((dorder[0], n)) for n in Bfilter[1:Aindex]]
        newedges2 = [MyEdge((dorder[1], n)) for n in Bfilter[Aindex+1:]]

        edges = newedges1 + newedges2

        return edges

    def add_road_segment_geo(self, edge, radius=1, epsilon=0.2):

        a, b, c1, c2 = self.__find_nodes_curbs(edge)
        m = mgh.bisect_angle(c1, b, c2, epsilon, radius=1)
        d1 = mgh.bisect_angle(a, b, m, epsilon, radius=radius)
        d2 = mgh.find_negative(d1, b)

        # figure out how the existing curb nodes connect to the new nodes
        new_d_edges = self.__find_d_connections(a, b, c1, c2, d1, d2)

        for e in new_d_edges:
            self.add_edge(e)

        # figure out how other involved parcels connect to the new nodes
        new_e_edges = self.__find_e_connections(a, b, c1, c2, d1, d2)

        for e in new_e_edges:
            self.add_edge(e)

        self.G.remove_node(b)
        self.roads_update = False

        return

# ###################################
#      PLOTTING FUNCTIONS
# ##################################

    def plot(self, **kwargs):
        plt.axes().set_aspect(aspect=1)
        plt.axis('off')
        edge_kwargs = kwargs.copy()
        nlocs = self.location_dict()
        edge_kwargs['label'] = "_nolegend"
        edge_kwargs['pos'] = nlocs
        nx.draw_networkx_edges(self.G, **edge_kwargs)
        node_kwargs = kwargs.copy()
        node_kwargs['label'] = self.name
        node_kwargs['pos'] = nlocs
        nodes = nx.draw_networkx_nodes(self.G, **node_kwargs)
        nodes.set_edgecolor('None')

    def plot_roads(self, master=None, update=False, parcel_labels=False,
                   title="", new_plot=True, new_road_color="blue",
                   new_road_width=6, old_node_size=30, old_road_width=8,
                   barriers=True):
        if new_plot:
            plt.figure()
        plt.axes().set_aspect(aspect=1)
        plt.axis('off')
        plt.title(title)
        nlocs = self.location_dict()

        if update:
            self.define_roads()
            self.define_interior_parcels()

        edge_colors = [new_road_color if e.road else 'black'
                       for e in self.myedges()]
        edge_width = [new_road_width if e.road else 1 for e in self.myedges()]
        # node_colors=['black' if n.road else 'black' for n in self.G.nodes()]
        node_colors = 'black'
        # node_sizes = [30 if n.road else 1 for n in self.G.nodes()]
        interior_graph = mgh.graphFromMyFaces(self.interior_parcels)

        # nx.draw_networkx(self.G,pos = nlocs, with_labels = False,
        #                  node_size = node_sizes, node_color= node_colors,
        #                  edge_color = edge_colors, width = edge_width)

        nx.draw_networkx_edges(self.G, pos=nlocs, with_labels=False,
                               node_size=1, node_color=node_colors,
                               edge_color=edge_colors, width=edge_width)

        nx.draw_networkx_edges(interior_graph.G, pos=nlocs, with_labels=False,
                               edge_color='red', node_color='red',
                               node_size=50, width=4)

        if parcel_labels is True:
            for i in range(0, len(self.inner_facelist)):
                plt.text(self.inner_facelist[i].centroid.x,
                         self.inner_facelist[i].centroid.y,
                         str(i), withdash=True)

        # plotting original road outline:

        if master:
            copy = master.copy()
            noffroad = [n for n in copy.G.nodes() if not n.road]
            for n in noffroad:
                    copy.G.remove_node(n)
            eoffroad = [e for e in copy.myedges() if not e.road]
            for e in eoffroad:
                copy.G.remove_edge(e.nodes[0], e.nodes[1])

            nx.draw_networkx(copy.G, pos=nlocs, with_labels=False,
                             node_size=old_node_size, node_color='black',
                             edge_color='black', width=old_road_width)

        if barriers:
            barrier_edges = [e for e in self.myedges() if e.barrier]
            if len(barrier_edges)>0:
                barGraph = mgh.graphFromMyEdges(barrier_edges)
                barGraph.plot(node_size=25, node_color='green', width=4,
                              edge_color='green')

    def plot_all_paths(self, all_paths, update=False):

        """ plots the shortest paths from all interior parcels to the road.
        Optional to update road geometery based on changes in network geometry.
        """

        plt.figure()
        if len(all_paths) == 0:
            self.plot_roads(update=update)
        else:
            Gs = []
            for p in all_paths:
                G = nx.subgraph(self.G, p)
                Gs.append(G)
            Gpaths = nx.compose_all(Gs, name="shortest paths")
            myGpaths = MyGraph(Gpaths)
            self.plot_roads(update=update)
            myGpaths.plot(edge_color='purple', width=6, node_size=1)


if __name__ == "__main__":
    master, n = mgh.testGraphLattice()

    S0 = master.copy()

    S0.define_roads()
    S0.define_interior_parcels()

    total_new_roads = 0
    new_roads = []

    edgesub = [e for e in S0.myedges()
               if e.nodes[0].y == 0 and e.nodes[1].y == 0]
    # barrieredges = [e for e in edgesub if e.nodes[1].y == 0]

    for e in edgesub:
        S0.remove_road_segment(e)
        e.barrier = True

    S0.define_interior_parcels()

    copy = S0.copy()

    S0.plot_roads(copy, update=False, new_plot=True)

    new_roads_i = mgh.build_all_roads(S0, alpha=2, wholepath=True,
                                     plot_intermediate=False)

    S0.plot_roads(copy, update=False, new_plot=True)
    # plt.title("Barrier Edges")

    plt.show()
