import shapefile
from matplotlib import pyplot as plt
import pylab
from operator import attrgetter
import numpy as np
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import timeit

import sys
sys.path.append(r'/Users/christa/Dropbox/SDI/Christa/topology_paper/python')
sys.path.append('C:\Users\F75VD\Dropbox\SDI\Christa\Topology_paper\python')
import my_graph as mg
import my_graph_helpers as mgh

    
def plot_roads(myG,update = False, parcel_labels = False):
    plt.figure(num ='roads', figsize=(8,4))
    plt.axes().set_aspect(aspect = 1)
    plt.axis('off')
    nlocs = myG.location_dict()
    
    
    if update:
        myG.trace_faces()
        myG.define_roads()
        myG.define_interior_parcels()
    
    edge_colors = ['black' if e.road else 'black' for e in myG.myedges()]
    edge_width = [6 if e.road else 1 for e in myG.myedges()]
    node_colors = ['black' if n.road else 'black' for n in myG.G.nodes()]
    #interior_graph = mgh.graphFromMyFaces(myG.interior_parcels)    
    
    #nx.draw_networkx(self.G,pos = nlocs, with_labels = False, node_size = node_sizes, node_color= node_colors, edge_color = edge_colors, width = edge_width)
    nx.draw_networkx_edges(myG.G,pos = nlocs, with_labels = False, node_size = 1, node_color= node_colors, edge_color = edge_colors, width = edge_width)  
    #nx.draw_networkx_edges(interior_graph.G, pos =  nlocs, with_labels = False, edge_color = 'red', node_color = 'red', node_size = 20, width = 4)
    
    if parcel_labels:
        for i in range(0,len(myG.inner_facelist)):
            plt.text(myG.inner_facelist[i].centroid.x,myG.inner_facelist[i].centroid.y, str(i), withdash = True)

    
    
    plt.savefig('parcels.png',dpi = 180)

        


if __name__ == "__main__":
    
    filename = "data/epworth_demo"
    # rezero = np.array([986850,216126])
    master = mgh.import_and_setup(0, filename, rezero=rezero)
    
    S0 = master.copy()
    S0.trace_faces()
    S0.define_roads()
    S0.define_interior_parcels()


    ## original setup
    
    
    plot_roads(S0,update = True, parcel_labels = False)
       
    S0 = master.copy().connected_components()
    
    S1s = [g.weak_dual().connected_components() for g in S0]
    S1 = [cc for duals in S1s for cc in duals]
    
    S2s = [g.weak_dual().connected_components() for g in S1]
    S2 = [cc for duals in S2s for cc in duals]
    
    S3s = [g.weak_dual().connected_components() for g in S2]
    S3 = [cc for duals in S3s for cc in duals]
    
    S4s = [g.weak_dual().connected_components() for g in S3]
    S4 = [cc for duals in S4s for cc in duals]
    
    
    colors= ['grey','purple','blue','red','orange','green']
    duals = [S0,S1] # , S2, S3,S4]
    width = [1,1,2,3,4]
    node_size = [1,20,30,40,60]
    plt.figure(num ='duals', figsize=(6,6))
    for i in range(0,len(duals)):
        for j in duals[i]:
            j.define_roads()
            j.define_interior_parcels()
            j.plot(node_size = node_size[i], node_color=colors[i], edge_color=colors[i], width =width[i] )
        
    plt.axes().set_aspect(aspect = 1)
    plt.axis('off')    
    # plt.savefig('duals.png',dpi = 180)
    
    
    
    S0[0].build_all_roads(2, plotting = True)
    
    S0[0].plot_roads(update = False)


    plt.show()
    
    

