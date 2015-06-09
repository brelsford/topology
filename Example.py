import numpy as np
import math
from matplotlib import pyplot as plt

import my_graph_helpers as mgh

"""
This example file demonstrates how to import a shapefile, finding and
 plotting the shortest distance of roads necessary to acheive universal road
 access for all parcels.

The shortest distance of roads algorithm is based on probablistic greedy search
so different runs will give slightly different answers.

 """


def run_once(filename, name=None):

    if name is None:
        name = filename

    original = mgh.import_and_setup(0, filename, threshold=1,
                                    name=name)

    block = original.copy()

    # define existing roads based on block geometery
    block.define_roads()

    # define interior parcels in the block based on existing roads
    block.define_interior_parcels()

    # plot roads, using the original, unedited version as master to color
    # original roads
    block.plot_roads(master=original)

    # finds roads to connect all interior parcels
    new_roads = mgh.build_all_roads(block, barriers=False)

    # plot new roads. original roads (black) defined by original graph.
    block.plot_roads(master=original, parcel_labels=False, new_plot=True)

    # red: interior parcels, bold black: original roads, blue: new roads,
    # green: barriers

    return new_roads


if __name__ == "__main__":

    # filename = "data/epworth_demo"
    # name = "epworth"

    filename = "data/capetown"
    name = "cape"

    run_once(filename, name)

    plt.show()
