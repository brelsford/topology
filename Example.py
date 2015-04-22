import numpy as np
import math
from matplotlib import pyplot as plt

import my_graph_helpers as mgh


def define_capetown_barriers(myG):
    """ based on rezero vector crezero = np.array([-31900, -3766370]) """
    be = [e for e in myG.myedges() if e.nodes[0].x < 146 and
          e.nodes[0].x > 25]
    be2 = [e for e in be if e.nodes[1].x < 146 and e.nodes[1].x > 25]

    be3 = [e for e in be2 if e.nodes[0].y < 20 and e.nodes[1].y < 20]
    todrop = [e for e in be3 if e.nodes[0].x > 25 and e.nodes[0].x < 75 and
              e.nodes[0].y > 13.4 and e.nodes[1].y > 13.4]

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
    be3 = [e for e in be2 if e.nodes[1].x > 100 and e.nodes[1].x < 113 and
           e.nodes[1].y > 119 and e.nodes[1].y < 140]

    return be+be3[0:-1]


if __name__ == "__main__":

    if True:  # make false in order to not re-import all the files.
        place = 'epworth'
        if place == 'epworth':
            filename = "data/epworth_demo"
            rezero = np.array([305680, 8022350])
            define_barriers = define_epworth_barriers
        if place == 'data/capetown':
            filename = "capetown"
            rezero = np.array([-31900, -3766370])
            define_barriers = define_capetown_barriers

        original = mgh.import_and_setup(0, filename, rezero=rezero,
                                        name=place+"_S0")
        original.define_roads()
        original.define_interior_parcels()
        barrier_edges = define_barriers(original)
        mgh.build_barriers(original, barrier_edges)

    # copy original, so that we can re-run from a clean base
    myG = original.copy()

    # plot roads, using the original, unedited version as master to color
    # original roads
    myG.plot_roads(master=original)

    # finds roads to connect all interior parcels
    new_roads = mgh.build_all_roads(myG)

    # plot new roads. original roads (black) defined by original graph.
    myG.plot_roads(master=original, parcel_labels=False, new_plot=True)

    # red: interior parcels, bold black: original roads, blue: new roads,
    # green: barriers

###############
# Dual graphs: pretty not important for stamen
###############

    stack = myG.stacked_duals()
    colors = ['grey', 'purple', 'blue', 'red', 'orange', 'yellow']
    width = [1, 1, 2, 4, 6]
    node_size = [2, 10, 25, 35, 45]
    myG.plot_weak_duals(stack=stack, colors=colors, width=width,
                        node_size=node_size)

    plt.show()
