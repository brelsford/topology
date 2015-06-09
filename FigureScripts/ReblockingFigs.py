# -*- coding: utf-8 -*-
"""
Created on Thu May  7 09:57:36 2015

Reblocking Figures

@author: Christa
"""
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
import pickle
from scipy.stats import norm

import my_graph_helpers as mgh


def new_length(block, a, r, plot=False):
    myG = block.copy()
    myG.define_roads()
    myG.define_interior_parcels()

    new_roads, bisect = mgh.build_all_roads(myG, barriers=False, alpha=a, 
                                            wholepath=True, quiet=True)
    if plot:
        myG.plot_roads(master=block, new_road_width=1.5, old_node_size=0.5,
                       old_road_width=2, base_width=0.5, barriers=False)
        plt.savefig('Figs/{0}_a{1}_r{2}.pdf'.format(myG.name, str(a), str(r)),
                    format='pdf')
    plt.close('all')
    return new_roads


def nice_histogram(a, x, bounds=None):
    num_bins = 80

    if bounds is None:
        bounds = [x.min(), x.max()]

    textstr1 = ' alpha={0} \n n={1}'.format(a, len(x))
    textstr2 = '\n mu={0:.2f} \n sigma={1:.2f}'.format(x.mean(), x.std())
    textstr = textstr1 + textstr2

    # the histogram of the data
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    n, bins, patches = plt.hist(x, num_bins, range=bounds, normed=1,
                                cumulative=True, facecolor='grey', alpha=0.5)

    # add a 'best fit' line
    y = norm(x.mean(), x.std()).cdf(bins)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Length of new roads')
    plt.ylabel('Probability')
    plt.title(r'$\alpha = {0}$'.format(a))
    plt.text(0.05, .8, textstr, transform=ax.transAxes)
    plt.savefig("histogram_alpha_"+str(a)+".pdf", format='pdf', pad_inches=0.5)


def nice_histogram_many(d):
    num_bins = 80
    plt.figure()
    for a in d.keys():
        n, bins, patches = plt.hist(d[a], num_bins, normed=1, cumulative=True,
                                    histtype='step', alpha=0.5,
                                    label="a = {}".format(a))
        plt.legend(loc='lower right')


if __name__ == "__main__":

    if False:
        filename = "data/capetown"
        place = "cape"
        crezero = np.array([-31900, -3766370])
        original = mgh.import_and_setup(0, filename, rezero=crezero,
                                        threshold=1, connected=False,
                                        name=place+"_S0")

    block = original.copy()
    block.define_roads()
    block.define_interior_parcels()

    myG = block.copy()

    # myG.plot_roads(master=block, new_road_width=1.5, old_node_size=0.5,
    #                old_road_width=2, base_width=0.5)
    # plt.savefig("Figs/cape_block.pdf", format='pdf',)

    alpha = [4, 16, 2]
    d = defaultdict(list)

    for a in alpha:
        r = 0
        print "alpha = {}".format(a)
        for r in range(0, 10):
            nr = new_length(block, a, r, plot=True)
            d[a].append(nr)
            print "r={}, alpha={}".format(r, a)

            r =+ 1
            pickle.dump(d, open("d_results2.p", "wb"))
            plt.close('all')

    nice_histogram_many(d)

    plt.show()
