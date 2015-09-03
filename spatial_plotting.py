from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import networkx as nx

import my_graph_helpers as mgh

def plot(myG, **kwargs):
    """ Basic spatial plotting for a graph."""

    
    plt.axes().set_aspect(aspect=1)
    plt.axis('off')
    edge_kwargs = kwargs.copy()
    nlocs = myG.location_dict()
    edge_kwargs['label'] = "_nolegend"
    edge_kwargs['pos'] = nlocs
    nx.draw_networkx_edges(myG.G, **edge_kwargs)
    node_kwargs = kwargs.copy()
    node_kwargs['label'] = myG.name
    node_kwargs['pos'] = nlocs
    nodes = nx.draw_networkx_nodes(myG.G, **node_kwargs)
    nodes.set_edgecolor('None')

def plot_roads(myG, master=None, update=False, title="", new_plot=True,
               new_road_color="blue",
               new_road_width=4, old_node_size=20, old_road_width=4,
               barriers=True, base_width=1):

    """Plot parcel roads, interior parcels, and barriers."""

    nlocs = myG.location_dict()

    nrc = new_road_color

    if update:
        myG.define_roads()
        myG.define_interior_parcels()

    if new_plot:
        plt.figure()
        plt.axes().set_aspect(aspect=1)
        plt.axis('off')

    plt.title = title

    edge_colors = [nrc if e.road
                   else 'green' if e.barrier
                   else 'red' if e.interior
                   else 'grey' for e in myG.myedges()]

    edge_width = [new_road_width if e.road
                  else 0.7*new_road_width if e.barrier
                  else 1 if e.interior
                  else 0.6 for e in myG.myedges()]

    node_colors = [nrc if n.road
                   else 'green' if e.barrier
                   else 'red' if n.interior
                   else 'grey' for n in myG.G.nodes()]

    node_sizes = [new_road_width**1.8 if n.road
                  else new_road_width**1.4 if n.barrier
                  else 1 if n.interior
                  else 0 for n in myG.G.nodes()]

    # plot current graph
    nx.draw_networkx_edges(myG.G, pos=nlocs, with_labels=False,
                      node_size=node_sizes, node_color=node_colors,
                      edge_color=edge_colors, width=edge_width)

    nodes = nx.draw_networkx_nodes(myG.G, pos=nlocs, with_labels=False,
                      node_size=node_sizes, node_color=node_colors,
                      edge_color=edge_colors, width=edge_width)
    nodes.set_edgecolor('None')


    # plot original roads
    if master:
        copy = master.copy()
        noffroad = [n for n in copy.G.nodes() if not n.road]
        for n in noffroad:
                copy.G.remove_node(n)
        eoffroad = [e for e in copy.myedges() if not e.road]
        for e in eoffroad:
            copy.G.remove_edge(e.nodes[0], e.nodes[1])

        nx.draw_networkx(copy.G, pos=nlocs, with_labels=False,
                          node_size=old_node_size, node_color='black',
                          edge_color='black', width=old_road_width)


def plot_all_paths(myG, all_paths, update=False):
    """ plots the shortest paths from all interior parcels to the road.
    Optional to update road geometery based on changes in network geometry.
    """

    plt.figure()
    if len(all_paths) == 0:
        myG.plot_roads(update=update)
    else:
        Gs = []
        for p in all_paths:
            G = nx.subgraph(self.G, p)
            Gs.append(G)
        Gpaths = nx.compose_all(Gs, name="shortest paths")
        myGpaths = MyGraph(Gpaths)
        myG.plot_roads(update=update)
        myGpaths.plot(edge_color='purple', width=6, node_size=1)

def plot_weak_duals(myG, stack=None, colors=None, width=None,
                    node_size=None, new_plot=True):
    """Given a list of weak dual graphs, plots them all. Has default colors
    node size, and line widths, but these can be added as lists.  Can only plot 7 duals."""

    if stack is None:
        duals = myG.stacked_duals()
    else:
        duals = stack

    if colors is None:
        colors = ['grey', 'black', 'blue', 'purple', 'red', 'orange',
                  'yellow']
    else:
        colors = colors

    if width is None:
        width = [0.5, 0.75, 1, 1.75, 2.25, 3, 3.5]
    else:
        width = width

    if node_size is None:
        node_size = [0.5, 6, 9, 12, 17, 25, 30]
    else:
        node_size = node_size

    if len(duals) > len(colors):
        warnings.warn("too many dual graphs to draw. simplify fig," +
                      " or add more colors")
    if new_plot:
        plt.figure()

    for i in range(0, len(duals)):
        for j in duals[i]:
            plot(j,node_size=node_size[i], node_color=colors[i],
                   edge_color=colors[i], width=width[i])
            # print "color = {0}, node_size = {1}, width = {2}".format(
            #       colors[i], node_size[i], width[i])

    plt.axes().set_aspect(aspect=1)
    plt.axis('off')


def plot_parcel_labels(graph):
    """Makes a plot of the graph, with parcel indices at the centroid.  """ 
    plt.figure()
    for p in graph.inner_facelist:
        index = graph.inner_facelist.index(p)
        #print "{} has index {}".format(p.centroid, index)
        #plt.plot([p.centroid.x], [p.centroid.y], 'g.')
        plt.text(p.centroid.x, p.centroid.y, str(index), {'size':3})
    plot(graph, node_size=0, node_color = 'grey', width = 0.5, edge_color='0.8')


def plot_cluster_mat(clustering_data, plotting_data, title, dmax=None, plot_dendro=True):
    """from http://nbviewer.ipython.org/github/OxanaSachenkova/
    hclust-python/blob/master/hclust.ipynb  First input matrix is used to
    define clustering order, second is the data that is plotted."""

    fig = plt.figure(figsize=(8, 8))
    # x ywidth height

    ax1 = fig.add_axes([0.05, 0.1, 0.2, 0.6])
    Y = linkage(clustering_data, method='single')
    Z1 = dendrogram(Y, orientation='right')  # adding/removing the axes
    ax1.set_xticks([])
    # ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3, 0.75, 0.6, 0.1])
    Z2 = dendrogram(Y)
    # ax2.set_xticks([])
    ax2.set_yticks([])

    # set up custom color map
    c = mcolors.ColorConverter().to_rgb
    seq = [c('navy'), c('mediumblue'), .1, c('mediumblue'),
           c('darkcyan'), .2, c('darkcyan'), c('darkgreen'), .3,
           c('darkgreen'), c('lawngreen'), .4,c('lawngreen'),c('yellow'),.5,
           c('yellow'), c('orange'), .7, c('orange'), c('red')]
    custommap = make_colormap(seq)

    # Compute and plot the heatmap
    if dmax is None:
        dmax = max([max(r) for r in plotting_data])
     
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])

    if not plot_dendro:
        fig = plt.figure(figsize=(8, 8))
        axmatrix = fig.add_axes([0.05, 0.1, 0.85, 0.8])

    idx1 = Z1['leaves']
    D = mgh.mat_reorder(plotting_data, idx1)
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=custommap,
                         vmin=0, vmax=dmax)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    h = 0.6
    if not plot_dendro:
        h = 0.8
    axcolor = fig.add_axes([0.91, 0.1, 0.02, h])
    plt.colorbar(im, cax=axcolor)
    ax2.set_title(title)
    if not plot_dendro:
        axmatrix.set_title(title)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


# ==============================================================================
# def plotly_traces(myG):
#     """myGraph to plotly trace   """
#
#     # add the edges as disconnected lines in a trace
#     edge_trace = Scatter(x=[], y=[], mode='lines',
#                          name='Parcel Boundaries',
#                          line=Line(color='grey', width=0.5))
#     road_trace = Scatter(x=[], y=[], mode='lines',
#                          name='Road Boundaries',
#                          line=Line(color='black', width=2))
#     interior_trace = Scatter(x=[], y=[], mode='lines',
#                              name='Interior Parcels',
#                              line=Line(color='red', width=2.5))
#     barrier_trace = Scatter(x=[], y=[], mode='lines',
#                             name='Barriers',
#                             line=Line(color='green', width=0.75))
#
#     for i in myG.connected_components():
#         for edge in i.myedges():
#             x0, y0 = edge.nodes[0].loc
#             x1, y1 = edge.nodes[1].loc
#             edge_trace['x'] += [x0, x1, None]
#             edge_trace['y'] += [y0, y1, None]
#             if edge.road:
#                 road_trace['x'] += [x0, x1, None]
#                 road_trace['y'] += [y0, y1, None]
#             if edge.interior:
#                 interior_trace['x'] += [x0, x1, None]
#                 interior_trace['y'] += [y0, y1, None]
#             if edge.barrier:
#                 barrier_trace['x'] += [x0, x1, None]
#                 barrier_trace['y'] += [y0, y1, None]
#
#     return [edge_trace, road_trace, interior_trace, barrier_trace]
#
#
# def plotly_graph(traces, filename=None, title=None):
#
#     """ use ply.iplot(fig,filename) after this function in ipython notebok to
#     show the resulting plotly figure inline, or url=ply.plot(fig,filename) to
#     just get url of resulting fig and not plot inline. """
#
#     if filename is None:
#         filename = "plotly_graph"
#     fig = Figure(data=Data(traces),
#                  layout=Layout(title=title, plot_bgcolor="rgb(217,217,217)",
#                                showlegend=True,
#                                xaxis=XAxis(showgrid=False, zeroline=False,
#                                            showticklabels=False),
#                                yaxis=YAxis(showgrid=False, zeroline=False,
#                                            showticklabels=False)))
#
#     # ply.iplot(fig, filename=filename)
#     # py.iplot(fig, filename=filename)
#
#     return fig, filename
#
