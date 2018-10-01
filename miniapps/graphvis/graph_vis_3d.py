"""
Graph visualization using matplotlib and networkx

Stephan Gelever
gelever1@llnl.gov
September 11, 2018
"""

import sys
import os

import numpy as np

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


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


def main(mat_filename, pos_filename, partition=[],
         vis_data=[], node_size=10, cols=1,
         dark=True, dual=False, log=False, color_map="jet"):
    """ Visualizes graph and position data

    Args:
        mat_filename:   Filename of matrix data
        pos_filename:   Filename of position data
        vis_data:       Data to visualize, else just show graph
        node_size:      Node size
        cols:           Columns in output
        dark:           Use dark background
        dual:           If input graph A is not square, use A^T A instead of AA^T
        log:            Use log of data values
        color_map:      Set colormap of data

    """
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

    if len(data) < 1:
        data.append(np.zeros(len(pos)))

    if len(partition) != len(pos):
        partition = np.zeros(len(pos))

    num_data = len(data) - 1
    num_vert = len(pos)
    num_edge = len(edges)

    x_node = np.zeros(num_vert)
    y_node = np.zeros(num_vert)
    z_node = np.zeros(num_vert)

    x_edge = []
    y_edge = []
    z_edge = []

    for key, value in pos.items():
        x_node[key], y_node[key] = value

    for edge in edges:
        x_edge += [pos[edge[0]][0], pos[edge[1]][0], None]
        y_edge += [pos[edge[0]][1], pos[edge[1]][1], None]
        z_edge += [0, 0, None]


    #for data_i in data:
    trace = []

    trace_vert = go.Scatter3d(x=x_node, y=y_node, z=z_node,
            mode='markers',
            marker=dict(
                size=1.000,
                color=partition,
                colorscale=color_map,
                opacity=1.0),
            hoverinfo='none',
            legendgroup="Graph",
            name="Graph"
            )

    trace_edge=go.Scatter3d(x=x_edge,
            y=y_edge,
            z=z_edge,
            mode='lines',
            line=dict(color='rgb(125,125,125)', width=1.5),
            hoverinfo='none',
            legendgroup="Graph",
            name=""
            )

    #trace_mesh = go.Mesh3d(x=x_node, y=y_node, z=z_node,
    #        color="blue", opacity=0.05, alphahull=-1,
    #        )
    # trace.extend([trace_vert, trace_edge, trace_mesh])

    trace.extend([trace_vert, trace_edge])

    for i, data_i in enumerate(data):
        x_bar = []
        y_bar = []
        z_bar = []
        color_bar = []

        x_edge = []
        y_edge = []
        z_edge = []

        for edge in edges:
            x_edge += [pos[edge[0]][0], pos[edge[1]][0], None]
            y_edge += [pos[edge[0]][1], pos[edge[1]][1], None]
            z_edge += [data_i[edge[0]], data_i[edge[1]], None]

        for key, value in pos.items():
            x_bar += [value[0], value[0], None]
            y_bar += [value[1], value[1], None]
            z_bar += [0, data_i[key], None]
            color_bar.extend([partition[key]] * 3)

        trace_vert = go.Scatter3d(x=x_node, y=y_node, z=data_i,
                mode='markers',
                marker=dict(
                    size=5,
                    color=partition,
                    colorscale=color_map,
                    opacity=0.9),
                hoverinfo="none",
                legendgroup="basis_" + str(i),
                name = "basis_" + str(i),
                visible="legendonly"
                )

        trace_edge=go.Scatter3d(x=x_edge,
                y=y_edge,
                z=z_edge,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=1.5),
                hoverinfo='none',
                legendgroup="basis_" + str(i),
                name = "",
                visible="legendonly"
                )

        trace_bar = go.Scatter3d(x=x_bar, y=y_bar, z=z_bar,
                mode='lines',
                line=dict(
                    width=05.0,
                    color=color_bar,
                    #color="black",
                    colorscale=color_map,
                    dash="solid"),
                hoverinfo="none",
                legendgroup="basis_" + str(i),
                name = "",
                visible="legendonly"
                )

        #trace.extend([trace_vert, trace_bar, trace_edge])
        trace.extend([trace_vert, trace_bar])
        #trace.extend([trace_bar])

    axis=dict(
            autorange=True,
            showgrid=True,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
            )

    layout = go.Layout(
            scene = dict(
                xaxis=axis,
                yaxis=axis,
                zaxis=axis
                )
            )

    plot(go.Figure(data=trace, layout=layout))


def parse_flag_val(argv, flag, dtype=str):
    """ Parse command line flag value

    Args:
        argv:           Command line arguments
        flag:           Flag to parse
        dtype:          Data type to return
    Returns:
        dtype           Flag value

    """
    flag_index = argv.index(flag)
    flag_val = argv[flag_index + 1]

    del argv[flag_index + 1]
    del argv[flag_index]

    return dtype(flag_val)


if __name__ == "__main__":
    node_size = 10.0
    cols = 2
    dark = True
    dual = False
    log = False
    color_map = "jet"
    vis_data = []
    partition = []

    if "-l" in sys.argv:
        dark = False
        sys.argv.remove("-l")

    if "-d" in sys.argv:
        dual = True
        sys.argv.remove("-d")

    if "-p" in sys.argv:
        partition = np.loadtxt(parse_flag_val(sys.argv, "-p", str))

    if "-log" in sys.argv:
        log = True
        sys.argv.remove("-log")

    if "-ns" in sys.argv:
        node_size = parse_flag_val(sys.argv, "-ns", float)

    if "-nc" in sys.argv:
        cols = parse_flag_val(sys.argv, "-nc", int)

    if "-cm" in sys.argv:
        color_map = parse_flag_val(sys.argv, "-cm", str)

    if (len(sys.argv) > 3):
        vis_data = sys.argv[3:]

    if len(sys.argv) < 3:
        print("Usage: mat_filename pos_filename [data] [-l] [-nc] [-d]")
    else:
        main(sys.argv[1], sys.argv[2], node_size=node_size,
             partition=partition, vis_data=vis_data,
             cols=cols, dark=dark, dual=dual, log=log,
             color_map=color_map)
