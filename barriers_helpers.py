# -*- coding: utf-8 -*-
"""
Specific barrier files.

@author: Christa
"""

import math
import numpy as np
import my_graph_helpers as mgh


def build_barriers(myG, edgelist):
    # assert isinstance(edgelist[0], mg.MyEdge), "{} is not and edge".
    # format(edgelist[0])
    for e in edgelist:
        if e in myG.myedges():
            myG.remove_road_segment(e)
            e.barrier = True


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


def load_capetown(barriers=False):
        filename = "data/capetown"
        place = "cape"
        crezero = np.array([-31900, -3766370])
        original = mgh.import_and_setup(0, filename, rezero=crezero,
                                        threshold=1, connected=False,
                                        name=place+"_S0")

        original.define_roads()
        original.define_interior_parcels()

        if barriers:
            capebar = define_capetown_barriers(original)
            build_barriers(original, capebar)

        return original


def load_epworth_science(barriers=False):
        filename = "data/epworth_demo"
        place = "epworth"
        erezero = np.array([305680, 8022350])
        original = mgh.import_and_setup(0, filename, rezero=erezero,
                                        threshold=1, connected=False,
                                        name=place+"_S0")

        original.define_roads()
        original.define_interior_parcels()
        if barriers:
            epbar = define_epworth_barriers(original)
            build_barriers(original, epbar)

        return original
        
        
def load_epworth_reblock(barriers=False):
        filename = "data/epworth_before"
        place = "epworth"
        erezero = np.array([305680, 8022350])
        original = mgh.import_and_setup(11, filename, rezero=erezero,
                                        threshold=1, connected=True,
                                        name=place+"_S0")

        original.define_roads()
        original.define_interior_parcels()
        if barriers:
            epbar = define_epworth_barriers(original)
            build_barriers(original, epbar)

        return original
        
if __name__ == "__main__":
    ep = load_epworth_reblock(barriers=False)
    #cape = load_capetown(barriers=False)
    
    
    ep.plot_roads(barriers=False)
    
    ep.plot_weak_duals()
    
    
    ## figure out how to include barriers code in with weak dual.  
