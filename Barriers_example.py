from matplotlib import pyplot as plt

import my_graph_helpers as mgh

"""
This shows a short snippet of code to (manually) define barriers for the
epworth block, build the barriers through the function "build_barriers", and
plot the results before and after construction of new roads.

 """


def define_epworth_barriers(myG):
    """these barriers are based on rezero vector
    np.array([305685.16  8022370.57])"""

    be = [e for e in myG.myedges() if e.nodes[0].x > 182 and
          e.nodes[1].x > 182]

    be2 = [e for e in myG.myedges() if e.nodes[0].x > 98 and
           e.nodes[0].x < 104 and e.nodes[0].y > 98 and e.nodes[0].y < 111]

    be3 = [e for e in myG.myedges() if e.nodes[1].x > 98 and
           e.nodes[1].x < 104 and e.nodes[1].y > 98 and e.nodes[1].y < 111]

    return be+be2+be3


if __name__ == "__main__":

    filename = "data/Epworth_demo"
    place = "epworth"

    original = mgh.import_and_setup(filename,
                                    threshold=0.5,
                                    byblock=True,
                                    name=place)
    original.define_roads()
    original.define_interior_parcels()

    barriers = define_epworth_barriers(original)
    mgh.build_barriers(barriers)

    original.plot_roads()

    mgh.build_all_roads(original, wholepath=True, barriers=True)

    original.plot_roads()
    plt.show()
