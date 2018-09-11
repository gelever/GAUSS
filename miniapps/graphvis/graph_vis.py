"""
Graph visualization using matplotlib and networkx

Stephan Gelever
gelever1@llnl.gov
September 11, 2018
"""

import sys
import os

import numpy as np

import scipy.sparse
import scipy.linalg
import subprocess

import networkx as nx
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import utilities


def compute_edge_color(edges, pos):
    """ Compute edge colors based on distance

    Args:
        edges:          List of edges
        pos:            List of vertex positions
    Returns:
        list            list of edge colors by distance

    """
    edge_color = [0 for _ in range(len(edges))]

    for i, edge in enumerate(edges):
        pos_0 = pos[edge[0]]
        pos_1 = pos[edge[1]]

        distance = [pos_0[0] - pos_1[0], pos_0[1] - pos_1[1]]
        distance = np.sqrt(
            (distance[0] * distance[0]) + (distance[1] * distance[1]))
        edge_color[i] = distance

    edge_max = max(edge_color)
    edge_color = [float(1.0 - (i / edge_max)) for i in edge_color]

    return edge_color


def parse_pos(filename):
    """ Parse a position file

    Data is a set of coordinates per node:
    x y
    x y
    ...

    Args:
        filename:       Name of file
    Returns:
        dict            dict of node positions

    """
    pos = {}

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split(" ")
            pos[i] = (float(line[0]), float(line[1]))

    return pos


def main(mat_filename, pos_filename, vis_data=[], node_size=40, cols=1, dark=True, dual=False, log=False, color_map="jet"):
    """ Visualizes graph and position data

    Args:
        mat_filename:   Filename of matrix data
        pos_filename:   Filename of position data
        vis_data:       Data to visualize, else just show graph
        node_size:      Data to visualize, else just show graph
        cols:           Columns in output
        dark:           Use dark background
        dual:           If input graph A is not square, use A^T A instead of AA^T
        log:            Use log of data values
        color_map:      Set colormap of data
    """
    if dark:
        plt.style.use('dark_background')

    mat = utilities.read_mat(mat_filename)

    if mat.shape[0] != mat.shape[1]:
        mat_t = mat.transpose()
        if not dual:
            mat = mat * mat_t
        else:
            mat = mat_t * mat

    pos = parse_pos(pos_filename)
    edges = utilities.make_edge_list(mat)
    data = [np.loadtxt(filename) for filename in vis_data]

    if log:
        data = [np.log(d) for d in data]

    print("Input: {0} Nodes, {1} Connections, {2} Positions".format(
        mat.shape[0], len(edges), len(pos)))

    num_data = len(vis_data)
    num_vert = len(pos)

    G = nx.Graph()
    G.add_nodes_from([i for i in range(num_vert)])
    G.add_edges_from(edges)

    cols = min(num_data + 1, cols)
    rows = int((num_data / cols) + 0.5)

    node_alpha = 1.0
    edge_alpha = 1.0
    edge_width = 0.8
    edge_color = "white" if dark else "silver"

    if num_data > 0:
        f, ax = plt.subplots(rows, cols)
        plt.set_cmap(color_map)

        for ax_i in np.ravel(ax):
            ax_i.axis('off')

        for i, data_i in enumerate(data):
            ax_i = np.ravel(ax)[i]
            nx.draw_networkx_nodes(G, pos, ax=ax_i, vmin=min(data_i), vmax=max(data_i),
                                   alpha=node_alpha, node_color=data_i, with_labels=False, node_size=node_size)
            nx.draw_networkx_edges(
                G, pos, edges, ax=ax_i, alpha=edge_alpha, width=edge_width, edge_color=edge_color)

        plt.show()

    else:
        edge_color = compute_edge_color(edges, pos)

        plt.figure()
        plt.set_cmap(color_map)

        nx.draw_networkx_edges(
            G, pos, edges, alpha=edge_alpha, width=edge_width, edge_color=edge_color)
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    node_size = 40.0
    cols = 2
    dark = True
    dual = False
    log = False
    color_map = "jet"
    vis_data = []

    if "-ns" in sys.argv:
        node_size = sys.argv[sys.argv.index("-ns") + 1]
        sys.argv.remove("-ns")
        sys.argv.remove(node_size)
        node_size = float(node_size)

    if "-l" in sys.argv:
        dark = False
        sys.argv.remove("-l")

    if "-d" in sys.argv:
        dual = True
        sys.argv.remove("-d")

    if "-log" in sys.argv:
        log = True
        sys.argv.remove("-log")

    if "-nc" in sys.argv:
        cols = sys.argv[sys.argv.index("-nc") + 1]
        sys.argv.remove("-nc")
        sys.argv.remove(cols)
        cols = int(cols)

    if "-cm" in sys.argv:
        color_map = sys.argv[sys.argv.index("-cm") + 1]
        sys.argv.remove("-cm")
        sys.argv.remove(color_map)

    if (len(sys.argv) > 3):
        vis_data = sys.argv[3:]

    if len(sys.argv) < 3:
        print("Usage: mat_filename pos_filename [data] [-l] [-nc] [-d]")
    else:
        main(sys.argv[1], sys.argv[2], node_size=node_size, vis_data=vis_data,
             cols=cols, dark=dark, dual=dual, log=log, color_map=color_map)
