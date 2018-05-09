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

    @brief MixedMatrix class
*/

#include "MixedMatrix.hpp"

namespace smoothg
{

template <>
MixedMatrix<std::vector<double>>::MixedMatrix(const Graph& graph)
    : edge_true_edge_(graph.edge_true_edge_),
      D_local_(MakeLocalD(graph.edge_true_edge_, graph.vertex_edge_local_)),
      W_local_(graph.W_local_),
      elem_dof_(graph.vertex_edge_local_)
{
    const int num_vertices = D_local_.Rows();

    M_elem_.resize(num_vertices);

    SparseMatrix edge_vertex = D_local_.Transpose();
    std::vector<double> weight_inv = graph.weight_local_;

    for (auto& i : weight_inv)
    {
        assert(std::fabs(i) > 1e-12);
        i = 1.0 / i;
    }

    for (int i = 0; i < num_vertices; ++i)
    {
        std::vector<int> edge_dofs = D_local_.GetIndices(i);

        int num_dofs = edge_dofs.size();

        M_elem_[i].resize(num_dofs);

        for (int j = 0; j < num_dofs; ++j)
        {
            M_elem_[i][j] = weight_inv[edge_dofs[j]] / edge_vertex.RowSize(edge_dofs[j]);
        }
    }

    agg_vertexdof_ = SparseIdentity(D_local_.Rows());
    agg_edgedof_ = D_local_;
    num_multiplier_dofs_ = agg_edgedof_.Cols();

    Init();
}

template <>
ParMatrix MixedMatrix<std::vector<double>>::ToPrimal() const
{
    assert(M_global_.Cols() == D_global_.Cols());
    assert(M_global_.Rows() == D_global_.Cols());

    ParMatrix MinvDT = D_global_.Transpose();
    MinvDT.InverseScaleRows(M_global_.GetDiag().GetDiag());

    ParMatrix A = D_global_.Mult(MinvDT);

    if (CheckW())
    {
        A = ParAdd(A, W_global_);
    }

    return A;
}

} // namespace smoothg
