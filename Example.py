from matplotlib import pyplot as plt

import my_graph_helpers as mgh

"""
This example file demonstrates how to import a shapefile, finding and
 plotting the shortest distance of roads necessary to acheive universal road
 access for all parcels.

The shortest distance of roads algorithm is based on probablistic greedy search
so different runs will give slightly different answers.

 """


def run_once(filename, name=None, byblock=True):

    if name is None:
        name = filename

    original = mgh.import_and_setup(filename, threshold=1,
                                    name=name)

    #  add code to sort blocklint by # interior parcels.
    blocklist = original.connected_components()

    print "This map has {} block(s).".format(len(blocklist))

    # plot original map.
    # plot roads, using the original, unedited version as master to color
    # original roads
    # block.plot_roads(master=original)

    map_roads = 0

    plt.figure()

    for original in blocklist[0:10]:

        # define existing roads based on block geometery
        original.define_roads()
        block = original.copy()

        # define interior parcels in the block based on existing roads
        block.define_interior_parcels()

        # finds roads to connect all interior parcels for a given block
        block_roads = mgh.build_all_roads(block, wholepath=True)
        map_roads = map_roads + block_roads

        block.plot_roads(master=original, new_plot=False)

    return map_roads


if __name__ == "__main__":

    # SINGLE SMALL BLOCK
    # filename = "data/epworth_demo"
    # name = "ep single"
    # byblock = True

    # MANY SMALL BLOCKS
    filename = "data/epworth_before"
    name = "ep many"
    byblock = True

    # ONE LARGE BLOCK
    # filename = "data/capetown"
    # name = "cape"
    # byblock = False

    run_once(filename, name, byblock=byblock)

    plt.show()
