"""
Utility functions for sparse matrices

Stephan Gelever
gelever1@llnl.gov
September 11, 2018
"""

import sys
import os

import numpy as np

import scipy.sparse
import scipy.linalg


def read_vertex_edge(filename):
    """ Reads a vertex edge matrix from file

    File is formated as follows:
        num vertices
        num edges
        indptr array
        indices array
        data array

    Args:
        filename: File holding matrix info

    Returns:
        scipy.sparse.csr_matrix:  Matrix read from file
                                  in CSR format

    """
    num_vertices = -1
    num_edges = -1
    indptr = []
    indices = []
    data = []

    with open(filename, 'r') as f:
        num_vertices = int(f.readline())
        num_edges = int(f.readline())

        for i in range(num_vertices + 1):
            indptr.append(int(f.readline()))

        for i in range(num_edges * 2):
            indices.append(int(f.readline()))

        for i in range(num_edges * 2):
            data.append(float(f.readline()))

    return scipy.sparse.csr_matrix((data, indices, indptr))


def write_vertex_edge(mat, filename="vertex_edge.txt"):
    """ Write a vertex edge matrix to file

    File is formated as follows:
        num vertices
        num edges
        indptr array
        indices array
        data array (of ones, not mat's data)

    Args:
        filename: File holding matrix info

    Returns:
        scipy.sparse.csr_matrix:  Matrix read from file
                                  in CSR format

    """
    with open(filename, 'w') as f:
        f.write(str(mat.shape[0]) + '\n')
        f.write(str(mat.shape[1]) + '\n')
        for i in [str(_) + '\n' for _ in mat.indptr]:
            f.write(i)

        for i in [str(_) + '\n' for _ in mat.indices]:
            f.write(i)

        for i in [str(_) + '\n' for _ in mat.data]:
            f.write("1.0\n")
            # f.write(i)


def read_csr(filename):
    """ Reads a csr matrix from file

    File is formated in space seperated list of coordinates:
        num rows
        num cols
        num non-zeros
        indptr array
        indices array
        data array

    Args:
        filename: File holding matrix info

    Returns:
        scipy.sparse.csr_matrix:  Matrix read from file
                                  in CSR format

    """
    rows = -1
    cols = -1
    nnz = -1

    indptr = []
    indices = []
    data = []

    with open(filename, 'r') as f:
        rows = int(f.readline())
        cols = int(f.readline())
        nnz = int(f.readline())

        for i in range(rows + 1):
            indptr.append(int(f.readline()))

        for i in range(nnz):
            indices.append(int(f.readline()))

        for i in range(nnz):
            data.append(float(f.readline()))

    return scipy.sparse.csr_matrix((data, indices, indptr))


def read_adj(filename):
    """ Reads a coordinate matrix from file

    File is formated in space seperated list of coordinates:
        i j val
        i j val
        ...

    Args:
        filename: File holding adjacency info

    Returns:
        scipy.sparse.csr_matrix:  Matrix read from file
                                  in CSR format

    """
    rows = []
    cols = []
    data = []

    with open(filename, 'r') as f:
        for line in f:
            vals = line.rstrip().split(' ')

            # TODO(gelever1): make this more elegant?
            if line.startswith("%") or line.startswith("#")\
                    or line.startswith("//") or line.startswith(" ")\
                    or len(vals) != 3:
                continue

            rows.append(int(vals[0]))
            cols.append(int(vals[1]))
            data.append(float(vals[2]))

    return scipy.sparse.coo_matrix((data, (rows, cols))).tocsr()


def read_edge_list(filename):
    """ Reads a edge list from file and creates edge vertex relationship

    File is formated in space seperated list of coordinates:
        i j 
        i j
        ...

    Args:
        filename: File holding adjacency info

    Returns:
        scipy.sparse.csr_matrix:  Matrix read from file
                                  in CSR format

    """
    rows = []
    cols = []
    data = []

    edges = []
    count = 0
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            vals = line.rstrip().split(' ')

            # TODO(gelever1): make this more elegant?
            if line.startswith("%") or line.startswith("#")\
                    or line.startswith("//") or line.startswith(" ")\
                    or len(vals) != 2:
                print("Skipping:", line)
                continue

            if int(vals[0]) != int(vals[1]):
                rows.append(count)
                cols.append(int(vals[0]))
                data.append(1.0)

                rows.append(count)
                cols.append(int(vals[1]))
                data.append(1.0)

                count += 1

    print(max(rows), max(cols))

    return scipy.sparse.coo_matrix((data, (rows, cols))).tocsr()


def read_system(basesys):
    """ Reads a csr matrix from directory

    Files are arrays of CSR format, ia, ja, sa

    Args:
        basesys: Directory holding csr matrix

    Returns:
        scipy.sparse.csr_matrix:  Matrix read from directory
                                  in CSR format

    """
    ia = []
    ja = []
    sa = []

    with open(os.path.join(basesys, "ia.txt"), 'r') as f:
        ia = np.ravel([int(_) for _ in f.readlines()])

    with open(os.path.join(basesys, "ja.txt"), 'r') as f:
        ja = np.ravel([int(_) for _ in f.readlines()])

    with open(os.path.join(basesys, "sa.txt"), 'r') as f:
        sa = np.ravel([float(_) for _ in f.readlines()])

    return scipy.sparse.csr_matrix((sa, ja, ia))


def save_system(A, directory="."):
    """ Write a csr matrix into directory

    Files are arrays of CSR format, ia, ja, sa

    Args:
        A:       Matrix to write
        basesys: Directory to write to


    """
    try:
        os.mkdir(directory)
    except FileExistsError:
        print("Writing into existing directory: %s" % directory)

    np.savetxt(os.path.join(directory, "ia.txt"),
               np.asarray(A.indptr, dtype=int), fmt="%d")
    np.savetxt(os.path.join(directory, "ja.txt"),
               np.asarray(A.indices, dtype=int), fmt="%d")
    np.savetxt(os.path.join(directory, "sa.txt"),
               np.asarray(A.data, dtype=float))
    np.savetxt(os.path.join(directory, "size.txt"),
               np.asarray([A.shape[0], A.nnz]), fmt="%d")


def make_edge_list(A):
    """ Creates an adjacency list from a sparse matrix

    For every row in A, created a list of connected vertices.
    Self connections are removed.

    Args:
        A (scipy.sparse.csr_matrix):  Matrix from which to create list

    Returns:
        list:   generated adjacency list

    """
    adj = []

    for row in range(A.shape[0]):
        for j in range(A.indptr[row], A.indptr[row + 1]):
            col = A.indices[j]

            if col > row:
                adj.append([row, col])

    return adj


def read_mat(filename):
    """ Try to parse a matrix file in many formats

    Args:
        filename:    Matrix file name

    Returns:
        scipy.sparse.csr_matrix:  Matrix read from file
                                  in CSR format

    """
    try:
        return read_vertex_edge(filename)
    except:
        try:
            return read_csr(filename)
        except:
            try:
                return read_system(filename)
            except:
                try:
                    return read_adj(filename)
                except:
                    return read_edge_list(filename)
