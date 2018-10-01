"""
Graph position generation using sfdp

Stephan Gelever
gelever1@llnl.gov
September 11, 2018
"""

import subprocess
import sys
import os

import utilities


def write_dot(edges, filename="graph.dot", k=1.0):
    """ Convert an edge list to dot file

    Args:
        edges:    List of edges
        filename: Output filename
        k:        Edge weight


    """
    with open(filename, 'w') as f:
        f.write("graph name {\n")

        for edge in edges:
            f.write("\t" + str(edge[0]) + " -- " +
                    str(edge[1]) + " [w=" + str(k) + "];\n")

        f.write("}")


def parse_dot(content):
    """ Parse a graph dot file

    Args:
        content:    Graph dot file contents

    Returns:
        dict:       Position dictionary


    """
    nodes = []
    for line in content.splitlines():
        if "node" not in line and " -- " not in line \
                and "graph" not in line and "Error" not in line \
                and "\t];" not in line and "overlap" not in line  \
                and "}" not in line:
            nodes.append(line)
    node_list = ""

    for node in nodes:
        node_list += "".join(node.split("\t"))

    node_list = node_list.split(";")

    node_list2 = []

    for node in node_list:
        node_list2.append(node.split(" "))
        node_list2[-1][-1] = node_list2[-1][-1].replace('[', '')
        node_list2[-1][-1] = node_list2[-1][-1].replace(']', '')
        node_list2[-1][-1] = node_list2[-1][-1].replace('=', ':')
        node_list2[-1][-1] = node_list2[-1][-1].replace(',pos', ' pos')
        node_list2[-1][-1] = node_list2[-1][-1].replace(',width', ' width')

    pos = {}
    for node in node_list2:
        if len(node) > 1:
            info = [float(i) for i in node[1].split(
                " ")[1].split(":")[-1].strip("\"").split(",")]
            pos[int(node[0])] = info

    return pos


def call_sfdp(filename="graph.dot", remove_overlap=False):
    """ Calls sfdp to generate node positions

    Args:
        filename:         Temporary dot graph file name
        remove_overlap:   Removes node overlaps, but can easily run out of memory on large problems

    Returns:
        string:           sfdp output


    """
    args = ["sfdp", filename]
    if remove_overlap:
        args.append("-Goverlap=prism")

    print(args)
    p = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = p.communicate()

    return stdout


def main(mat_filename, output_filename="output.pos", dual=False, overlap=False):
    """ Generates coordinate file from sfdp

    Args:
        mat_filename:   Filename of matrix in data directory
        output_filename:   Filename of matrix in data directory

    """
    mat = utilities.read_mat(mat_filename)

    if mat.shape[0] != mat.shape[1]:
        mat_t = mat.transpose()
        if not dual:
            mat = mat * mat_t
        else:
            mat = mat_t * mat

    edges = utilities.make_edge_list(mat)

    tmp_graph = "graph.dot"
    write_dot(edges, tmp_graph)
    sfdp = call_sfdp(tmp_graph, overlap)

    pos = parse_dot(sfdp)

    with open(output_filename, 'w') as f:
        for p in sorted(pos.items(), key=lambda x: x[0]):
            f.write(str(p[1][0]) + " " + str(p[1][1]) + "\n")


if __name__ == "__main__":
    overlap = False
    dual = False
    output_filename = "output.pos"

    if "-o" in sys.argv:
        output_filename = sys.argv[sys.argv.index("-o") + 1]
        sys.argv.remove("-o")
        sys.argv.remove(output_filename)

    if "-d" in sys.argv:
        dual = True
        sys.argv.remove("-d")

    if "-overlap" in sys.argv:
        overlap = True
        sys.argv.remove("-overlap")

    if len(sys.argv) is not 2:
        print("Usage: matrix_file [-o output_file] [-d]")
    else:
        main(sys.argv[1], output_filename=output_filename,
             dual=dual, overlap=overlap)
