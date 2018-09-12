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

    @brief Contains Graph class
*/

#include "Graph.hpp"

namespace smoothg
{

Graph::Graph(MPI_Comm comm, const SparseMatrix& vertex_edge_global,
             const std::vector<int>& part_global,
             const std::vector<double>& weight_global,
             const SparseMatrix& W_block_global)
    : global_vertices_(vertex_edge_global.Rows()),
      global_edges_(vertex_edge_global.Cols())
{
    assert(static_cast<int>(part_global.size()) == vertex_edge_global.Rows());

    int myid;
    MPI_Comm_rank(comm, &myid);

    SparseMatrix agg_vert = MakeAggVertex(part_global);
    SparseMatrix proc_agg = MakeProcAgg(comm, agg_vert, vertex_edge_global);

    SparseMatrix proc_vert = proc_agg.Mult(agg_vert);
    SparseMatrix proc_edge = proc_vert.Mult(vertex_edge_global);

    proc_edge.SortIndices();

    vertex_map_ = proc_vert.GetIndices(myid);
    std::vector<int> edge_map = proc_edge.GetIndices(myid);

    vertex_edge_local_ = vertex_edge_global.GetSubMatrix(vertex_map_, edge_map);
    vertex_edge_local_ = 1.0;

    int nvertices_local = proc_vert.RowSize(myid);
    part_local_.resize(nvertices_local);

    for (int i = 0; i < nvertices_local; ++i)
    {
        part_local_[i] = part_global[vertex_map_[i]];
    }

    ShiftPartition(part_local_);

    edge_true_edge_ = MakeEdgeTrueEdge(comm, proc_edge, edge_map);

    ParMatrix edge_true_edge_T = edge_true_edge_.Transpose();
    edge_edge_ = edge_true_edge_.Mult(edge_true_edge_T);

    MakeLocalWeight(edge_map, weight_global);
    MakeLocalW(W_block_global);
}

void Graph::MakeLocalWeight(const std::vector<int>& edge_map,
                            const std::vector<double>& global_weight)
{
    int num_edges = vertex_edge_local_.Cols();

    weight_local_.resize(num_edges);

    if (static_cast<int>(global_weight.size()) == edge_true_edge_.GlobalCols())
    {
        assert(static_cast<int>(edge_map.size()) == num_edges);

        for (int i = 0; i < num_edges; ++i)
        {
            assert(std::fabs(global_weight[edge_map[i]]) > 1e-14);
            weight_local_[i] = std::fabs(global_weight[edge_map[i]]);
        }
    }
    else
    {
        std::fill(std::begin(weight_local_), std::end(weight_local_), 1.0);
    }

    const SparseMatrix& edge_offd = edge_edge_.GetOffd();

    assert(edge_offd.Rows() == num_edges);

    for (int i = 0; i < num_edges; ++i)
    {
        if (edge_offd.RowSize(i))
        {
            weight_local_[i] *= 2.0;
        }
    }
}

void Graph::MakeLocalW(const SparseMatrix& W_global)
{
    if (W_global.Rows() > 0)
    {
        W_local_ = W_global.GetSubMatrix(vertex_map_, vertex_map_);
        W_local_ *= -1.0;
    }
}

Graph::Graph(SparseMatrix vertex_edge_local, ParMatrix edge_true_edge,
             std::vector<int> part_local,
             std::vector<double> weight_local,
             SparseMatrix W_block_local)
    : part_local_(std::move(part_local)),
      vertex_edge_local_(std::move(vertex_edge_local)),
      edge_true_edge_(std::move(edge_true_edge)),
      edge_edge_(edge_true_edge_.Mult(edge_true_edge_.Transpose())),
      weight_local_(std::move(weight_local)),
      W_local_(std::move(W_block_local)),
      global_edges_(edge_true_edge_.GlobalCols())
{
    int num_vertices = vertex_edge_local_.Rows();
    int num_edges = vertex_edge_local_.Cols();

    MPI_Comm comm = edge_true_edge_.GetComm();

    auto vertex_starts = parlinalgcpp::GenerateOffsets(comm, num_vertices);

    global_vertices_ = vertex_starts.back();

    vertex_map_.resize(num_vertices);
    std::iota(std::begin(vertex_map_), std::end(vertex_map_), vertex_starts[0]);

    if (static_cast<int>(weight_local_.size()) != num_edges)
    {
        MakeLocalWeight();
    }

    W_local_ *= -1.0;
}

Graph::Graph(const Graph& other) noexcept
    : vertex_map_(other.vertex_map_),
      part_local_(other.part_local_),
      vertex_edge_local_(other.vertex_edge_local_),
      edge_true_edge_(other.edge_true_edge_),
      edge_edge_(other.edge_edge_),
      weight_local_(other.weight_local_),
      W_local_(other.W_local_),
      global_vertices_(other.global_vertices_),
      global_edges_(other.global_edges_)
{

}

Graph::Graph(Graph&& other) noexcept
{
    swap(*this, other);
}

Graph& Graph::operator=(Graph other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(Graph& lhs, Graph& rhs) noexcept
{
    std::swap(lhs.vertex_map_, rhs.vertex_map_);
    std::swap(lhs.part_local_, rhs.part_local_);

    swap(lhs.vertex_edge_local_, rhs.vertex_edge_local_);
    swap(lhs.edge_true_edge_, rhs.edge_true_edge_);
    swap(lhs.edge_edge_, rhs.edge_edge_);

    swap(lhs.weight_local_, rhs.weight_local_);
    swap(lhs.W_local_, rhs.W_local_);

    std::swap(lhs.global_vertices_, rhs.global_vertices_);
    std::swap(lhs.global_edges_, rhs.global_edges_);
}

} // namespace smoothg

