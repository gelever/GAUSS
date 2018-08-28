/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

    @brief Contains GraphSpace class
*/

#ifndef GRAPHSPACE_HPP
#define GRAPHSPACE_HPP

#include "Utilities.hpp"
#include "Graph.hpp"

namespace smoothg
{

/**
   @brief Collection of data per level
*/
struct GraphSpace
{
    ParMatrix vertex_vdof;
    ParMatrix vertex_edof;
    ParMatrix vertex_bdof;
    ParMatrix edge_edof;

    ParMatrix agg_vertexdof;
    ParMatrix face_facedof;
};

inline
GraphSpace FineGraphSpace(const Graph& graph)
{
    MPI_Comm comm = graph.edge_true_edge_.GetComm();

    int num_vertices = graph.vertex_edge_local_.Rows();
    int num_edges = graph.vertex_edge_local_.Cols();

    return
    {
        {comm, SparseIdentity(num_vertices)},
        {comm, graph.vertex_edge_local_},
        {comm, SparseMatrix(num_vertices, num_edges)},
        {comm, SparseIdentity(num_edges)},
        {comm, SparseIdentity(num_vertices)},
        {comm, SparseIdentity(num_edges)},
    };
}


} // namespace smoothg

#endif /* GRAPHSPACE_HPP */
