# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:07:52 2015

testing inside out construction

@author: Christa
"""

import my_graph_helpers as mgh
import my_graph as mg

simpleG = mgh.testGraphLattice(5)
simpleG.name = "L0"
simpleduals = simpleG.stacked_duals()

myG = simpleG.copy()

myG.define_roads()
myG.define_interior_parcels()
myG.plot_roads(update=True)


result, depth = mgh.form_equivalence_classes(myG)


mgh.build_all_roads(myG, alpha=16, barriers=False, wholepath=True, 
                    plot_intermediate=True)
