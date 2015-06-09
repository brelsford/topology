# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 09:32:32 2015

@author: Christa
"""



def define_capetown_barriers(myG):
    """ these barriers are based on rezero vector
    np.array([-31900, -3766370]) """
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
    """these barriers are based on rezero vector np.array([305680, 8022350])"""
    be = [e for e in myG.myedges() if e.nodes[0].x > 187 and
          e.nodes[1].x > 187]

    be2 = [e for e in myG.myedges() if e.nodes[0].x > 100 and
           e.nodes[0].x < 113 and e.nodes[0].y > 119 and e.nodes[0].y < 140]
    be3 = [e for e in be2 if e.nodes[1].x > 100 and e.nodes[1].x < 113 and
           e.nodes[1].y > 119 and e.nodes[1].y < 140]

    return be+be3[0:-1]
