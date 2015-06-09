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

    new_roads = mgh.build_all_roads(myG, barriers=False, alpha=a,
                                    wholepath=True, strict_greedy=False,
                                    quiet=False, outsidein=False)
    if plot:
        myG.plot_roads(master=block, new_road_width=1.5, old_node_size=0.5,
                       old_road_width=2, base_width=0.5, barriers=False)
        plt.savefig('Figs/{}_a{}_r{}.pdf'.format(myG.name, str(a), str(r)),
                    format='pdf')
    # plt.close('all')
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
    plt.savefig("epworth_histogram_alpha_"+str(a)+".pdf", format='pdf',
                pad_inches=0.5)


def nice_histogram_many(d, keys, xval):
    num_bins = 80
    plt.figure()
    plt.vlines(x=xval, ymin=0, ymax=1, linewidth=3, colors='orange',
               linestyles='dashed')
    for a in keys:
        n, bins, patches = plt.hist(d[a], num_bins, normed=1, cumulative=True,
                                    histtype='stepfilled', alpha=0.5,
                                    label="a = {}".format(a))
        plt.legend(loc='lower right')

    plt.savefig("Figs/full_histogram_epworth_outside_in.pdf", format='pdf')


if __name__ == "__main__":

    if True:
        filename = "data/epworth_before"
        place = "epworth"
        crezero = np.array([-31900, -3766370])
        erezero = np.array([305680, 8022350])
        original = mgh.import_and_setup(filename, rezero=erezero,
                                        component=None,
                                        threshold=1, connected=True,
                                        name=place+"_S0")

    blocklist = original.connected_components()

    sevens = []
    comp = 0

    for g in blocklist:
        comp += 1
        g.define_roads()
        g.define_interior_parcels()
        if len(g.interior_parcels) == 7:
            sevens.append(g)

    block = sevens[3]

    myG = block.copy()

    myG.plot_roads(master=block, new_road_width=1.5, old_node_size=0.5,
                   old_road_width=2, base_width=0.5)
    plt.savefig("Figs/epworth_block.pdf", format='pdf',)

    alpha = [0.5, 1, 2, 4, 16, 32, 64]
    alpha = [128, 1000] # , 64]
    d = defaultdict(list)

    r = 0

    for r in range(0, 10):
        for a in alpha:
            print "r={}, alpha={}".format(r, a)
            nr = new_length(block, a, r, plot=False)
            print "total new roads {}".format(nr)
            d[a].append(nr)
            pickle.dump(d, open("epworth_alpha2.p", "wb"))
        plt.close('all')
        r += 1

    strict = mgh.build_all_roads(block.copy(), barriers=False, alpha=16,
                                 wholepath=True, strict_greedy=True,
                                 quiet=False, outsidein=True)

    nice_histogram_many(d, [1000, 128, 64,32,16], strict)

    plt.show()
