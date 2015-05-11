# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:57:36 2015

Reblocking Figures

@author: Christa
"""
import numpy as np
import math
from matplotlib import pyplot as plt

import my_graph_helpers as mgh


if __name__ == "__main__":

    if False:
        filename = "data/epworth_before"
        place = "epworth"
        rezero = np.array([305680, 8022350])
        original = mgh.import_and_setup(1, filename, rezero=rezero,
                                        connected=True, name=place+"_S0")

    block = original.connected_components()[7]
    block.define_roads()
    block.define_interior_parcels()

    # copy original, so that we can re-run from a clean base
    myG = block.copy()
    
    # finds roads to connect all interior parcels
    
    alpha = [0.5,1,2,4,8]
    for a in alpha:
        myG.define_roads()
        myG.define_interior_parcels()
        new_roads = mgh.build_all_roads(myG, barriers=False, alpha=a)
        myG.plot_roads(master=block)
        plt.savefig("test"+str(a)+".pdf", format='pdf')

    plt.show()

    print "done"
    
    ## cc3, 7 11 potential
