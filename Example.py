import numpy as np
import math
from matplotlib import pyplot as plt

#import sys
#sys.path.append(r'/Users/christa/Dropbox/SDI/Christa/topology_paper/python')
#sys.path.append('C:\Users\F75VD\Dropbox\SDI\Christa\Topology_paper\python')

import my_graph as mg
import my_graph_helpers as mgh


def define_capetown_barriers(myG):
    """ based on rezero vector crezero = np.array([-31900, -3766370]) """
    be = [e for e in myG.myedges() if e.nodes[0].x < 146 and
          e.nodes[0].x > 25]
    be2 = [e for e in be if e.nodes[1].x < 146 and e.nodes[1].x > 25]

    be3 = [e for e in be2 if e.nodes[0].y < 20 and e.nodes[1].y < 20]
    todrop = [e for e in be3 if e.nodes[0].x > 25 and e.nodes[0].x < 75
              and e.nodes[0].y > 13.4 and e.nodes[1].y > 13.4]

    for e in be3:
        if abs(e.rads) > math.pi/4:
            todrop.append(e)

    be4 = [e for e in be3 if e not in todrop]

    return be4


def define_epworth_barriers(myG):
    """ erezero = np.array([305680, 8022350])  """
    be = [e for e in myG.myedges() if e.nodes[0].x > 187 and
          e.nodes[1].x > 187]

    be2 = [e for e in myG.myedges() if e.nodes[0].x > 100 and
           e.nodes[0].x < 113 and e.nodes[0].y > 119 and e.nodes[0].y < 140]
    be3 = [e for e in be2 if e.nodes[1].x > 100 and e.nodes[1].x < 113
           and e.nodes[1].y > 119 and e.nodes[1].y < 140]

    return be+be3[0:-1]


if __name__ == "__main__":
    # make false in order to not re-import all the files.  The rest of the code
    # just runs based on original
    if True:
        place = 'epworth'
        if place == 'epworth':
            filename = "data/epworth_demo"
            rezero = np.array([305680, 8022350])
            define_barriers = define_epworth_barriers
        if place == 'data/capetown':
            filename = "capetown"
            rezero = np.array([-31900, -3766370])
            define_barriers = define_capetown_barriers

        original = mgh.import_and_setup(0, filename, rezero=rezero)
        original.define_roads()
        original.define_interior_parcels()

    myG = original.copy()

    barrier_edges = define_barriers(myG)
    mgh.build_barriers(myG, barrier_edges)

    barriers = myG.copy()

    myG.plot_roads(master=barriers)

    # makes a new graph taking into accuont the barriers as defined above
    barGraph = mgh.graphFromMyEdges(barrier_edges)

    new_roads = mgh.build_all_roads(myG, barriers, bisect=False)

    myG.plot_roads(barriers, parcel_labels=False, new_plot=True)

    stacks = myG.stacked_duals()

    plt.show()
    
    # red: interior parcels, bold black: original roads, blue: new roads,
    # green: barriers
    
