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

    @brief GraphCoarsen class
*/

#include <unordered_map>
#include "GraphCoarsen.hpp"

namespace smoothg
{
GraphCoarsen::GraphCoarsen(const Graph& graph, const MixedMatrix& mgl,
                           int max_evects, double spect_tol)
    : GraphCoarsen(GraphTopology(graph), mgl, max_evects, spect_tol)
{

}


bool fine = true;

GraphCoarsen::GraphCoarsen(GraphTopology gt, const MixedMatrix& mgl,
                           int max_evects, double spect_tol)
    : gt_(std::move(gt)),
      max_evects_(max_evects), spect_tol_(spect_tol),
      vertex_targets_(gt_.agg_ext_edge_.Rows()),
      edge_targets_(gt_.face_edge_.Rows()),
      agg_ext_sigma_(gt_.agg_ext_edge_.Rows()),
      B_potential_(gt_.agg_ext_edge_.Rows()),
      D_trace_sum_(gt_.agg_ext_edge_.Rows()),
      D_trace_(gt_.agg_ext_edge_.Rows()),
      F_potential_(gt_.agg_ext_edge_.Rows())
{
    MPI_Comm comm = mgl.GlobalD().GetComm();

    ParMatrix v_vdof(comm, mgl.vertex_vdof);
    ParMatrix v_edof(comm, mgl.vertex_edof);
    ParMatrix v_bdof(comm, mgl.vertex_bdof);
    ParMatrix e_edof(comm, mgl.edge_edof);

    ParMatrix true_edge_edge = gt_.edge_true_edge_.Transpose();
    ParMatrix no_bub = gt_.agg_ext_edge_.Mult(true_edge_edge).Mult(e_edof).Mult(mgl.EdgeTrueEdge());
    ParMatrix with_bub = gt_.agg_ext_vertex_.Mult(v_bdof).Mult(mgl.EdgeTrueEdge());

    agg_ext_edof_ = parlinalgcpp::ParAdd(no_bub, with_bub);
    agg_ext_vdof_ = gt_.agg_ext_vertex_.Mult(v_vdof);

    ParMatrix permute_v = MakeExtPermutation(gt_.agg_ext_vertex_.Mult(v_vdof));
    ParMatrix permute_e = MakeExtPermutation(agg_ext_edof_);

    ParMatrix permute_v_T = permute_v.Transpose();
    ParMatrix permute_e_T = permute_e.Transpose();

    ParMatrix M_ext_global = permute_e.Mult(mgl.GlobalM().Mult(permute_e_T));
    ParMatrix D_ext_global = permute_v.Mult(mgl.GlobalD().Mult(permute_e_T));

    ParMatrix face_perm_edge = gt_.face_edge_.Mult(e_edof).Mult(mgl.EdgeTrueEdge()).Mult(permute_e_T);

    int marker_size = std::max(permute_v.Rows(), permute_e.Rows());
    col_marker_.resize(marker_size, -1);

    SparseMatrix agg_edgedof = gt_.agg_edge_local_.Mult(mgl.edge_edof);
    SparseMatrix agg_bubbledof = gt_.agg_vertex_local_.Mult(mgl.vertex_bdof);

    agg_edgedof_ = Add(1.0, agg_edgedof, 1.0, agg_bubbledof);
    agg_vertexdof_ = gt_.agg_vertex_local_.Mult(mgl.vertex_vdof);
    face_edgedof_ = gt_.face_edge_local_.Mult(mgl.edge_edof);

    ComputeVertexTargets(M_ext_global, D_ext_global);
    ComputeEdgeTargets(mgl, face_perm_edge);

    BuildFaceCoarseDof();
    BuildAggBubbleDof();
    BuildPvertex();
    BuildPedge(mgl);
}

GraphCoarsen::GraphCoarsen(const GraphCoarsen& other) noexcept
    : gt_(other.gt_),
      max_evects_(other.max_evects_),
      spect_tol_(other.spect_tol_),
      P_edge_(other.P_edge_),
      P_vertex_(other.P_vertex_),
      face_cdof_(other.face_cdof_),
      agg_bubble_dof_(other.agg_bubble_dof_),
      vertex_targets_(other.vertex_targets_),
      edge_targets_(other.edge_targets_),
      agg_ext_sigma_(other.agg_ext_sigma_),
      // Tmp stuff
      agg_vertexdof_(other.agg_vertexdof_),
      agg_edgedof_(other.agg_edgedof_),
      face_edgedof_(other.face_edgedof_),
      agg_ext_vdof_(other.agg_ext_vdof_),
      agg_ext_edof_(other.agg_ext_edof_),
      // End tmp stuff
      col_marker_(other.col_marker_),
      B_potential_(other.B_potential_),
      D_trace_sum_(other.D_trace_sum_),
      D_trace_(other.D_trace_),
      F_potential_(other.F_potential_)
{

}

GraphCoarsen::GraphCoarsen(GraphCoarsen&& other) noexcept
{
    swap(*this, other);
}

GraphCoarsen& GraphCoarsen::operator=(GraphCoarsen other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(GraphCoarsen& lhs, GraphCoarsen& rhs) noexcept
{
    swap(lhs.gt_, rhs.gt_);

    std::swap(lhs.max_evects_, rhs.max_evects_);
    std::swap(lhs.spect_tol_, rhs.spect_tol_);

    swap(lhs.P_edge_, rhs.P_edge_);
    swap(lhs.P_vertex_, rhs.P_vertex_);
    swap(lhs.face_cdof_, rhs.face_cdof_);
    swap(lhs.agg_bubble_dof_, rhs.agg_bubble_dof_);

    swap(lhs.vertex_targets_, rhs.vertex_targets_);
    swap(lhs.edge_targets_, rhs.edge_targets_);
    swap(lhs.agg_ext_sigma_, rhs.agg_ext_sigma_);

    swap(lhs.col_marker_, rhs.col_marker_);

    swap(lhs.B_potential_, rhs.B_potential_);
    swap(lhs.D_trace_sum_, rhs.D_trace_sum_);
    swap(lhs.D_trace_, rhs.D_trace_);
    swap(lhs.F_potential_, rhs.F_potential_);

    // Temp Stuff
    swap(lhs.agg_vertexdof_, rhs.agg_vertexdof_);
    swap(lhs.agg_edgedof_, rhs.agg_edgedof_);
    swap(lhs.face_edgedof_, rhs.face_edgedof_);

    swap(lhs.agg_ext_vdof_, rhs.agg_ext_vdof_);
    swap(lhs.agg_ext_edof_, rhs.agg_ext_edof_);
    // End Temp Stuff
}

void GraphCoarsen::ComputeVertexTargets(const ParMatrix& M_ext_global,
                                        const ParMatrix& D_ext_global)
{
    const SparseMatrix& M_ext = M_ext_global.GetDiag();
    const SparseMatrix& D_ext = D_ext_global.GetDiag();

    int num_aggs = gt_.agg_ext_edge_.Rows();

    DenseMatrix evects;
    LocalEigenSolver eigs(max_evects_, spect_tol_);

    DenseMatrix DT_evect;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> edge_dofs_ext = GetExtDofs(agg_ext_edof_, agg);
        std::vector<int> vertex_dofs_ext = GetExtDofs(agg_ext_vdof_, agg);

        std::vector<int> vertex_dofs_local = agg_vertexdof_.GetIndices(agg);

        if (edge_dofs_ext.size() == 0)
        {
            vertex_targets_[agg] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        SparseMatrix M_sub = M_ext.GetSubMatrix(edge_dofs_ext, edge_dofs_ext, col_marker_);
        SparseMatrix D_sub = D_ext.GetSubMatrix(vertex_dofs_ext, edge_dofs_ext, col_marker_);

        eigs.BlockCompute(M_sub, D_sub, evects);

        if (evects.Cols() > 1)
        {
               DenseMatrix evects_ortho = evects.GetCol(1, evects.Cols());

               DT_evect.SetSize(D_sub.Cols(), evects_ortho.Cols());
               D_sub.MultAT(evects_ortho, DT_evect);

               DenseMatrix MinvDT_evect(DT_evect.Rows(), DT_evect.Cols());

               SparseSolver Minv(M_sub);

               // TODO(gelever1): define this in SparseSolver::Mult(const DenseMatrix)
               // or something of the sort
               for (int i = 0; i < DT_evect.Cols(); ++i)
               {
                   auto in_vect = DT_evect.GetColView(i);
                   auto out_vect = MinvDT_evect.GetColView(i);
                   Minv.Mult(in_vect, out_vect);
               }

               agg_ext_sigma_[agg] = std::move(MinvDT_evect);
        }
        else
        {
            agg_ext_sigma_[agg].SetSize(M_sub.Rows(), 0);
        }

        DenseMatrix evects_restricted = RestrictLocal(evects, col_marker_,
                vertex_dofs_ext, vertex_dofs_local);

        VectorView first_vect = evects_restricted.GetColView(0);
        vertex_targets_[agg] = smoothg::Orthogonalize(evects_restricted, first_vect, 1, max_evects_);
    }
}

std::vector<std::vector<DenseMatrix>> GraphCoarsen::CollectSigma(const SparseMatrix& face_edgedof)
{
    SharedEntityComm<DenseMatrix> sec_sigma(gt_.face_true_face_);

    int num_faces = gt_.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_dofs = face_edgedof.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        int total_vects = 0;
        int col_count = 0;

        for (auto agg : neighbors)
        {
            total_vects += agg_ext_sigma_[agg].Cols();
        }

        DenseMatrix face_sigma(face_dofs.size(), total_vects);

        for (auto agg : neighbors)
        {
            if (agg_ext_sigma_[agg].Cols() > 0)
            {
                std::vector<int> edge_dofs_ext = GetExtDofs(agg_ext_edof_, agg);

                DenseMatrix face_restrict = RestrictLocal(agg_ext_sigma_[agg], col_marker_,
                                                          edge_dofs_ext, face_dofs);

                face_sigma.SetCol(col_count, face_restrict);
                col_count += face_restrict.Cols();
            }
        }

        assert(col_count == total_vects);

        sec_sigma.ReduceSend(face, std::move(face_sigma));
    }

    return sec_sigma.Collect();
}

std::vector<std::vector<Vector>> GraphCoarsen::CollectConstant(const MixedMatrix& mgl)
{
    const auto& fine_constant = mgl.constant_vect_;

    SharedEntityComm<Vector> sec_constant(gt_.face_true_face_);

    int num_faces = gt_.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);
        std::vector<double> constant_data;

        for (auto agg : neighbors)
        {
            std::vector<int> agg_vertices = agg_vertexdof_.GetIndices(agg);
            auto sub_vect = fine_constant.GetSubVector(agg_vertices);

            constant_data.insert(std::end(constant_data), std::begin(sub_vect),
                                   std::end(sub_vect));
        }

        sec_constant.ReduceSend(face, Vector(std::move(constant_data)));
    }

    return sec_constant.Collect();
}

std::vector<std::vector<SparseMatrix>> GraphCoarsen::CollectD(const MixedMatrix& mgl)
{
    const auto& D_local = mgl.LocalD();

    SharedEntityComm<SparseMatrix> sec_D(gt_.face_true_face_);

    int num_faces = gt_.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> vertex_ext_dofs;
        std::vector<int> edge_ext_dofs = face_edgedof_.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = agg_edgedof_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));

            std::vector<int> agg_vertices = agg_vertexdof_.GetIndices(agg);
            vertex_ext_dofs.insert(std::end(vertex_ext_dofs), std::begin(agg_vertices),
                                   std::end(agg_vertices));
        }

        SparseMatrix D_face = D_local.GetSubMatrix(vertex_ext_dofs, edge_ext_dofs, col_marker_);

        sec_D.ReduceSend(face, std::move(D_face));
    }

    return sec_D.Collect();
}

std::vector<std::vector<SparseMatrix>> GraphCoarsen::CollectM(const SparseMatrix& M_local)
{
    SharedEntityComm<SparseMatrix> sec_M(gt_.face_true_face_);

    int num_faces = gt_.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> edge_ext_dofs = face_edgedof_.GetIndices(face);
        std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);

        for (auto agg : neighbors)
        {
            std::vector<int> agg_edges = agg_edgedof_.GetIndices(agg);
            edge_ext_dofs.insert(std::end(edge_ext_dofs), std::begin(agg_edges),
                                 std::end(agg_edges));
        }

        SparseMatrix M_face = M_local.GetSubMatrix(edge_ext_dofs, edge_ext_dofs, col_marker_);

        sec_M.ReduceSend(face, std::move(M_face));
    }

    return sec_M.Collect();
}

void GraphCoarsen::ComputeEdgeTargets(const MixedMatrix& mgl,
                                      const ParMatrix& face_perm_edge)
{
    const SparseMatrix& face_edge = face_perm_edge.GetDiag();

    auto shared_sigma = CollectSigma(face_edge);
    auto shared_constant = CollectConstant(mgl);
    auto shared_M = CollectM(mgl.LocalM());
    auto shared_D = CollectD(mgl);

    const SparseMatrix& face_shared = gt_.face_face_.GetOffd();

    SharedEntityComm<DenseMatrix> sec_face(gt_.face_true_face_);
    DenseMatrix collected_sigma;

    int num_faces = gt_.face_edge_local_.Rows();

    for (int face = 0; face < num_faces; ++face)
    {
        int num_face_edges = face_edge.RowSize(face);

        if (!sec_face.IsOwnedByMe(face))
        {
            edge_targets_[face].SetSize(num_face_edges, 0);
            continue;
        }

        if (num_face_edges == 1)
        {
            edge_targets_[face] = DenseMatrix(1, 1, {1.0});
            continue;
        }

        const auto& face_M = shared_M[face];
        const auto& face_D = shared_D[face];
        const auto& face_sigma = shared_sigma[face];
        const auto& face_constant = shared_constant[face];

        linalgcpp::HStack(face_sigma, collected_sigma);

        bool shared = face_shared.RowSize(face) > 0;

        // TODO(gelever1): resolve this copy.  (Types must match so face_? gets copied and promoted
        // to rvalue).
        const auto& M_local = shared ? CombineM(face_M, num_face_edges) : face_M[0];
        const auto& D_local = shared ? CombineD(face_D, num_face_edges) : face_D[0];
        const auto& constant_local = shared ? CombineConstant(face_constant) : face_constant[0];
        const int split = shared ? face_D[0].Rows() : GetSplit(face);


        GraphEdgeSolver solver(M_local, D_local);
        Vector one_neg_one = MakeOneNegOne(constant_local, split);

        if (std::fabs(constant_local.Mult(one_neg_one)) > 1e-12)
        {
            printf("%d constnat * one neg one : %.8e\n", MyId(), constant_local.Mult(one_neg_one) );
        }

        Vector pv_sol = solver.Mult(one_neg_one);
        VectorView pv_sigma(pv_sol.begin(), num_face_edges);

        edge_targets_[face] = Orthogonalize(collected_sigma, pv_sigma, 0, max_evects_);
    }

    sec_face.Broadcast(edge_targets_);

    ScaleEdgeTargets(mgl);
}

void GraphCoarsen::ScaleEdgeTargets(const MixedMatrix& mgl)
{
    const auto& D_local = mgl.LocalD();
    int num_faces = gt_.face_edge_.Rows();

    Vector one;
    Vector oneD;

    for (int face = 0; face < num_faces; ++face)
    {
        DenseMatrix& edge_traces(edge_targets_[face]);

        if (edge_traces.Cols() < 1)
        {
            continue;
        }

        int agg = gt_.face_agg_local_.GetIndices(face)[0];

        std::vector<int> vertices = agg_vertexdof_.GetIndices(agg);
        std::vector<int> face_dofs = face_edgedof_.GetIndices(face);

        SparseMatrix D_transfer = D_local.GetSubMatrix(vertices, face_dofs, col_marker_);

        mgl.constant_vect_.GetSubVector(vertices, one);

        oneD.SetSize(D_transfer.Cols());
        D_transfer.MultAT(one, oneD);

        VectorView pv_trace = edge_traces.GetColView(0);

        double oneDpv = oneD.Mult(pv_trace);
        double beta = (oneDpv < 0) ? -1.0 : 1.0;
        oneDpv *= beta;

        pv_trace /= oneDpv;

        int num_traces = edge_traces.Cols();

        for (int k = 1; k < num_traces; ++k)
        {
            VectorView trace = edge_traces.GetColView(k);
            double alpha = oneD.Mult(trace);

            trace.Sub(alpha * beta, pv_trace);
        }
    }
}

SparseMatrix GraphCoarsen::CombineM(const std::vector<SparseMatrix>& face_M, int num_face_edges) const
{
    assert(face_M.size() == 2);

    int size = face_M[0].Rows() + face_M[1].Rows() - num_face_edges;
    int offset = face_M[0].Rows() - num_face_edges;

    CooMatrix M_coo(size, size);
    M_coo.Reserve(face_M[0].nnz() + face_M[1].nnz());

    const auto& M_0_indptr = face_M[0].GetIndptr();
    const auto& M_0_indices = face_M[0].GetIndices();
    const auto& M_0_data = face_M[0].GetData();
    const auto& M_0_rows = face_M[0].Rows();

    for (int i = 0; i < M_0_rows; ++i)
    {
        for (int j = M_0_indptr[i]; j < M_0_indptr[i + 1]; ++j)
        {
            M_coo.Add(i, M_0_indices[j], M_0_data[j]);
        }
    }

    const auto& M_1_indptr = face_M[1].GetIndptr();
    const auto& M_1_indices = face_M[1].GetIndices();
    const auto& M_1_data = face_M[1].GetData();
    const auto& M_1_rows = face_M[1].Rows();

    for (int i = 0; i < M_1_rows; ++i)
    {
        for (int j = M_1_indptr[i]; j < M_1_indptr[i + 1]; ++j)
        {
            int col = M_1_indices[j];

            if (i < num_face_edges && col < num_face_edges)
            {
                M_coo.Add(i, M_1_indices[j], M_1_data[j]);
            }
            else
            {
                M_coo.Add(i + offset, M_1_indices[j] + offset, M_1_data[j]);
            }
        }
    }

    return M_coo.ToSparse();
}

SparseMatrix GraphCoarsen::CombineD(const std::vector<SparseMatrix>& face_D,
                                   int num_face_edges) const
{
    assert(face_D.size() == 2);

    int rows = face_D[0].Rows() + face_D[1].Rows();
    int cols = face_D[0].Cols() + face_D[1].Cols() - num_face_edges;

    std::vector<int> indptr = face_D[0].GetIndptr();
    indptr.insert(std::end(indptr), std::begin(face_D[1].GetIndptr()) + 1,
                  std::end(face_D[1].GetIndptr()));

    int row_start = face_D[0].Rows() + 1;
    int row_end = indptr.size();
    int nnz_offset = face_D[0].nnz();

    for (int i = row_start; i < row_end; ++i)
    {
        indptr[i] += nnz_offset;
    }

    std::vector<int> indices = face_D[0].GetIndices();
    indices.insert(std::end(indices), std::begin(face_D[1].GetIndices()),
                   std::end(face_D[1].GetIndices()));

    int col_offset = face_D[0].Cols() - num_face_edges;
    int nnz_end = indices.size();

    for (int i = nnz_offset; i < nnz_end; ++i)
    {
        if (indices[i] >= num_face_edges)
        {
            indices[i] += col_offset;
        }
    }

    std::vector<double> data = face_D[0].GetData();
    data.insert(std::end(data), std::begin(face_D[1].GetData()),
                std::end(face_D[1].GetData()));

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        rows, cols);
}

Vector GraphCoarsen::CombineConstant(const std::vector<Vector>& face_constant) const
{
    assert(face_constant.size() == 2);

    int split = face_constant[0].size();
    int size = face_constant[0].size() + face_constant[1].size();

    Vector vect(size);

    std::copy(std::begin(face_constant[0]), std::end(face_constant[0]), std::begin(vect));
    std::copy(std::begin(face_constant[1]), std::end(face_constant[1]), std::begin(vect) + split);

    return vect;
}

Vector GraphCoarsen::MakeOneNegOne(int size, int split) const
{
    assert(size >= 0);
    assert(split >= 0);

    Vector vect(size);

    for (int i = 0; i < split; ++i)
    {
        vect[i] = 1.0 / split;
    }

    for (int i = split; i < size; ++i)
    {
        vect[i] = -1.0 / (size - split);
    }

    return vect;
}

Vector GraphCoarsen::MakeOneNegOne(const Vector& constant, int split) const
{
    assert(split >= 0);

    int size = constant.size();

    Vector vect(size);

    double v1_sum = 0.0;
    double v2_sum = 0.0;

    for (int i = 0; i < split; ++i)
    {
        v1_sum += constant[i] * constant[i];
    }

    for (int i = split; i < size; ++i)
    {
        v2_sum += constant[i] * constant[i];
    }

    double c1 = 1.0 / split;
    double c2 = -c1 * (v1_sum / v2_sum);

    for (int i = 0; i < split; ++i)
    {
        vect[i] = c1 * constant[i];
    }

    for (int i = split; i < size; ++i)
    {
        vect[i] = c2 * constant[i];
    }

    return vect;
}

int GraphCoarsen::GetSplit(int face) const
{
    std::vector<int> neighbors = gt_.face_agg_local_.GetIndices(face);
    assert(neighbors.size() >= 1);
    int agg = neighbors[0];

    return gt_.agg_vertex_local_.RowSize(agg);
}

void GraphCoarsen::BuildFaceCoarseDof()
{
    int num_faces = gt_.face_edge_.Rows();

    std::vector<int> indptr(num_faces + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_faces; ++i)
    {
        indptr[i + 1] = indptr[i] + edge_targets_[i].Cols();
    }

    int num_traces = indptr.back();

    std::vector<int> indices(num_traces);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::vector<double> data(num_traces, 1.0);

    face_cdof_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                              num_faces, num_traces);
}

void GraphCoarsen::BuildAggBubbleDof()
{
    int num_aggs = vertex_targets_.size();

    std::vector<int> indptr(num_aggs + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        assert(vertex_targets_[i].Cols() >= 1);

        indptr[i + 1] = indptr[i] + vertex_targets_[i].Cols() - 1;
    }

    int num_traces = SumCols(edge_targets_);
    int num_bubbles = indptr.back();

    std::vector<int> indices(num_bubbles);
    std::iota(std::begin(indices), std::end(indices), num_traces);

    std::vector<double> data(num_bubbles, 1.0);

    agg_bubble_dof_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                                   num_aggs, num_traces + num_bubbles);
}

void GraphCoarsen::BuildPvertex()
{
    const SparseMatrix& agg_vertex = agg_vertexdof_;
    int num_aggs = vertex_targets_.size();
    int num_vertices = agg_vertex.Cols();

    std::vector<int> indptr(num_vertices + 1);
    indptr[0] = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> vertices = agg_vertex.GetIndices(i);
        int num_coarse_dofs = vertex_targets_[i].Cols();

        for (auto vertex : vertices)
        {
            indptr[vertex + 1] = num_coarse_dofs;
        }
    }

    for (int i = 0; i < num_vertices; ++i)
    {
        indptr[i + 1] += indptr[i];
    }

    int nnz = indptr.back();
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);

    int coarse_dof_counter = 0;

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> fine_dofs = agg_vertex.GetIndices(i);
        int num_fine_dofs = fine_dofs.size();
        int num_coarse_dofs = vertex_targets_[i].Cols();

        const DenseMatrix& target_i = vertex_targets_[i];

        for (int j = 0; j < num_fine_dofs; ++j)
        {
            int counter = indptr[fine_dofs[j]];

            for (int k = 0; k < num_coarse_dofs; ++k)
            {
                indices[counter] = coarse_dof_counter + k;
                data[counter] = target_i(j, k);

                counter++;
            }
        }

        coarse_dof_counter += num_coarse_dofs;
    }

    P_vertex_ = SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                             num_vertices, coarse_dof_counter);
}

int ComputeNNZ(const GraphTopology& gt, const SparseMatrix& agg_bubble_dof,
               const SparseMatrix& face_cdof)
{
    const SparseMatrix& agg_face = gt.agg_face_local_;
    //const SparseMatrix& agg_edge = agg_edgedof_;
    //const SparseMatrix& face_edge = face_edgedof_;
    // TODO(gelever1): This is wrong cus dofs not considered
    const SparseMatrix& agg_edge = gt.agg_edge_local_;
    const SparseMatrix& face_edge = gt.face_edge_local_;

    int num_aggs = agg_edge.Rows();
    int num_faces = face_edge.Rows();

    int nnz = 0;
    for (int agg = 0; agg < num_aggs; ++agg)
    {
        int edge_dofs = agg_edge.RowSize(agg);
        int bubble_dofs = agg_bubble_dof.RowSize(agg);

        std::vector<int> faces = agg_face.GetIndices(agg);

        for (auto face : faces)
        {
            int face_coarse_dofs = face_cdof.RowSize(face);
            nnz += edge_dofs * face_coarse_dofs;
        }

        nnz += edge_dofs * bubble_dofs;
    }

    for (int face = 0; face < num_faces; ++face)
    {
        int face_fine_dofs = face_edge.RowSize(face);
        int face_coarse_dofs = face_cdof.RowSize(face);

        nnz += face_fine_dofs * face_coarse_dofs;
    }

    return nnz;
}

void GraphCoarsen::BuildPedge(const MixedMatrix& mgl)
{
    const SparseMatrix& agg_face = gt_.agg_face_local_;
    const SparseMatrix& agg_edge = agg_edgedof_;
    const SparseMatrix& face_edge = face_edgedof_;
    const SparseMatrix& agg_vertex = agg_vertexdof_;

    int num_aggs = agg_edge.Rows();
    int num_faces = face_edge.Rows();
    int num_edges = agg_edge.Cols();
    int num_coarse_dofs = agg_bubble_dof_.Cols();

    CooMatrix P_edge(num_edges, num_coarse_dofs);
    P_edge.Reserve(ComputeNNZ(gt_, agg_bubble_dof_, face_cdof_));

    DenseMatrix bubbles;
    DenseMatrix trace_ext;
    Vector one;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> faces = agg_face.GetIndices(agg);
        std::vector<int> edge_dofs = agg_edge.GetIndices(agg);
        std::vector<int> vertex_dofs = agg_vertex.GetIndices(agg);
        std::vector<int> bubble_dofs = agg_bubble_dof_.GetIndices(agg);

        SparseMatrix M = mgl.LocalM().GetSubMatrix(edge_dofs, edge_dofs, col_marker_);
        SparseMatrix D = mgl.LocalD().GetSubMatrix(vertex_dofs, edge_dofs, col_marker_);

        mgl.constant_vect_.GetSubVector(vertex_dofs, one);

        GraphEdgeSolver solver(M, D);

        for (auto face : faces)
        {
            std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(face);
            std::vector<int> face_fine_dofs = face_edge.GetIndices(face);

            SparseMatrix D_transfer = mgl.LocalD().GetSubMatrix(vertex_dofs, face_fine_dofs, col_marker_);
            DenseMatrix D_trace = D_transfer.Mult(edge_targets_[face]);

            D_trace_sum_[agg].push_back(D_trace.GetColView(0).Mult(vertex_targets_[agg].GetColView(0)));

            OrthoConstant(D_trace, one);

            DenseMatrix F_potential;
            solver.Mult(D_trace, trace_ext, F_potential);
            OrthoConstant(F_potential, one);

            D_trace_[agg].push_back(std::move(D_trace));
            F_potential_[agg].push_back(std::move(F_potential));

            P_edge.Add(edge_dofs, face_coarse_dofs, trace_ext);
        }

        solver.OffsetMult(1, vertex_targets_[agg], bubbles, B_potential_[agg]);
        P_edge.Add(edge_dofs, bubble_dofs, bubbles);

        OrthoConstant(B_potential_[agg], one);
    }

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> face_fine_dofs = face_edge.GetIndices(face);
        std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(face);

        P_edge.Add(face_fine_dofs, face_coarse_dofs, -1.0, edge_targets_[face]);
    }

    //P_edge.EliminateZeros(1e-10);
    P_edge_ = P_edge.ToSparse();
}

SparseMatrix GraphCoarsen::BuildAggCDofVertex() const
{
    //TODO(gelever1): do this proper
    SparseMatrix agg_cdof_vertex = agg_vertexdof_.Mult(P_vertex_);
    agg_cdof_vertex.SortIndices();

    return agg_cdof_vertex;
}

SparseMatrix GraphCoarsen::BuildAggCDofEdge() const
{
    int num_aggs = gt_.agg_ext_edge_.Rows();
    int num_traces = face_cdof_.Cols();
    int num_cdofs = P_edge_.Cols();

    std::vector<int> indptr(num_aggs + 1);
    indptr[0] = 0;

    std::vector<int> indices;

    int bubble_counter = 0;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        int num_bubbles_i = vertex_targets_[agg].Cols() - 1;
        for (int i = 0; i < num_bubbles_i; ++i)
        {
            indices.push_back(num_traces + bubble_counter + i);
        }

        std::vector<int> faces = gt_.agg_face_local_.GetIndices(agg);

        for (auto face : faces)
        {
            std::vector<int> face_cdofs = face_cdof_.GetIndices(face);

            for (auto dof : face_cdofs)
            {
                indices.push_back(dof);
            }
        }

        indptr[agg + 1] = indices.size();

        bubble_counter += num_bubbles_i;
    }

    std::vector<double> data(indices.size(), 1.0);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_aggs, num_cdofs);
}

ParMatrix GraphCoarsen::BuildEdgeTrueEdge() const
{
    int num_faces = face_cdof_.Rows();
    int num_traces = face_cdof_.Cols();
    int num_coarse_dofs = P_edge_.Cols();

    const auto& ftf_diag = gt_.face_true_face_.GetDiag();

    int num_true_dofs = num_coarse_dofs - num_traces;

    for (int i = 0; i < num_faces; ++i)
    {
        if (ftf_diag.RowSize(i) > 0)
        {
            num_true_dofs += face_cdof_.RowSize(i);
        }
    }

    MPI_Comm comm = gt_.face_true_face_.GetComm();
    auto cface_starts = parlinalgcpp::GenerateOffsets(comm, num_coarse_dofs);
    const auto& face_starts = gt_.face_true_face_.GetRowStarts();

    SparseMatrix face_cdof_expand(face_cdof_.GetIndptr(), face_cdof_.GetIndices(),
                                  face_cdof_.GetData(), num_faces, num_coarse_dofs);
    ParMatrix face_cdof_d(comm, face_starts, cface_starts, std::move(face_cdof_expand));

    ParMatrix cface_cface = parlinalgcpp::RAP(gt_.face_face_, face_cdof_d);

    const SparseMatrix& cface_cface_offd = cface_cface.GetOffd();
    const std::vector<int>& cface_cface_colmap = cface_cface.GetColMap();

    std::vector<int> offd_indptr(num_coarse_dofs + 1);
    offd_indptr[0] = 0;

    int offd_nnz = 0;

    for (int i = 0; i < num_coarse_dofs; ++i)
    {
        if (cface_cface_offd.RowSize(i) > 0)
        {
            offd_nnz++;
        }

        offd_indptr[i + 1] = offd_nnz;
    }

    std::vector<int> offd_indices(offd_nnz);
    std::vector<double> offd_data(offd_nnz, 1.0);

    const auto& face_cdof_indptr = face_cdof_.GetIndptr();
    const auto& face_cdof_indices = face_cdof_.GetIndices();

    int col_count = 0;

    for (int i = 0; i < num_faces; ++i)
    {
        if (gt_.face_face_.GetOffd().RowSize(i) > 0)
        {
            int first_dof = face_cdof_indices[face_cdof_indptr[i]];

            std::vector<int> face_cdofs = cface_cface_offd.GetIndices(first_dof);
            assert(static_cast<int>(face_cdofs.size()) == face_cdof_.RowSize(i));

            for (auto cdof : face_cdofs)
            {
                offd_indices[col_count++] = cdof;
            }
        }
    }

    assert(col_count == offd_nnz);
    assert(col_count == static_cast<int>(cface_cface_colmap.size()));

    SparseMatrix d_td_d_diag = SparseIdentity(num_coarse_dofs);
    SparseMatrix d_td_d_offd(std::move(offd_indptr), std::move(offd_indices),
                             std::move(offd_data), num_coarse_dofs, col_count);

    ParMatrix d_td_d(comm, cface_starts, cface_starts,
                     std::move(d_td_d_diag), std::move(d_td_d_offd),
                     cface_cface_colmap);

    return MakeEntityTrueEntity(d_td_d);
}

SparseMatrix GraphCoarsen::BuildCoarseD() const
{
    int num_aggs = gt_.agg_ext_edge_.Rows();
    int total_traces = face_cdof_.Cols();

    int counter = 0;

    CooMatrix D_coarse(P_vertex_.Cols(), P_edge_.Cols());

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        std::vector<int> faces = gt_.agg_face_local_.GetIndices(agg);
        int num_faces = faces.size();

        for (int j = 0; j < num_faces; ++j)
        {
            double val = -1.0 * D_trace_sum_[agg][j];
            std::vector<int> face_coarse_dofs = face_cdof_.GetIndices(faces[j]);

            D_coarse.Add(counter + agg, face_coarse_dofs[0], val);
        }

        int num_bubbles_i = vertex_targets_[agg].Cols() - 1;

        for (int j = 0; j < num_bubbles_i; ++j)
        {
            D_coarse.Add(counter + agg + 1 + j, total_traces + counter + j, 1.0);
        }

        counter += num_bubbles_i;
    }

    return D_coarse.ToSparse();
}

/// Helper function to build fine-level aggregate sub-M
/// corresponding to dofs on a face
void BuildAggregateFaceM(const MixedMatrix& mgl, int agg,
                         const std::vector<int>& edge_dofs_on_face,
                         const SparseMatrix& vertex_agg,
                         const SparseMatrix& edge_vertex,
                         std::vector<int>& col_marker,
                         DenseMatrix& M_local)
{
    const auto& M_fine_elem = mgl.GetElemM();
    const auto& partition = vertex_agg.GetIndices();

    int fine_size = edge_dofs_on_face.size();
    M_local.SetSize(fine_size);
    M_local = 0.0;

    for (int i = 0; i < fine_size; ++i)
    {
        int edge_dof = edge_dofs_on_face[i];
        auto verts = edge_vertex.GetIndices(edge_dof);
        int vert = (partition[verts[0]] == agg ) ? verts[0] : verts[1];

        std::vector<int> fine_edges = mgl.GetElemDof().GetIndices(vert);

        SetMarker(col_marker, fine_edges);

        int index = col_marker[edge_dof];
        assert(index >= 0);

        M_local(i, i) = M_fine_elem[vert](index, index);

        ClearMarker(col_marker, fine_edges);
    }
}

void GraphCoarsen::BuildAggFaceM(const MixedMatrix& mgl, int face, int agg,
                      const SparseMatrix& vertex_agg,
                      const SparseMatrix& edge_vertex,
                      std::vector<int>& col_marker,
                      DenseMatrix& M_local) const
{
    const auto& M_fine_elem = mgl.GetElemM();
    const auto& partition = vertex_agg.GetIndices();

    std::vector<int> fine_edges = gt_.face_edge_local_.GetIndices(face);
    std::vector<int> fine_dofs = face_edgedof_.GetIndices(face);

    std::unordered_map<int, int> test;

    int num_fine_edge = fine_edges.size();
    int num_fine_dof = fine_dofs.size();

    for (int i = 0; i < num_fine_dof; ++i)
    {
        test[fine_dofs[i]] = i;
    }

    M_local.SetSize(num_fine_dof);
    M_local = 0.0;

    DenseMatrix sub;

    SetMarker(col_marker, fine_dofs);

    for (int i = 0; i < num_fine_edge; ++i)
    {
        int fine_edge = fine_edges[i];
        std::vector<int> edge_dofs = mgl.edge_edof.GetIndices(fine_edge);
        std::vector<int> verts = edge_vertex.GetIndices(fine_edge);

        int vert = (partition[verts[0]] == agg ) ? verts[0] : verts[1];

        //printf("Face M : %d %d %d %d\n", agg, i, vert, fine_edge);

        /*
        std::vector<int> elem_dofs = mgl.GetElemDof().GetIndices(vert);

        std::vector<int> M_indices;
        std::vector<int> agg_indices;

        for (int j = 0; j < elem_dofs.size(); ++j)
        {
            int dof = elem_dofs[j];

            if (test.find(dof) != test.end())
            {
                M_indices.push_back(test[dof]);
                agg_indices.push_back(j);
            }
        }

        auto submat = M_fine_elem[vert].GetSubMatrix(agg_indices, agg_indices);

        M_local.AddSubMatrix(M_indices, M_indices, submat);
        */

        ///*
        std::vector<int> fine_dofs = mgl.GetElemDof().GetIndices(vert);
        SetMarker(col_marker, fine_dofs);

        int num_dofs = edge_dofs.size();

        sub.SetSize(num_dofs);
        sub = 0.0;

        std::vector<int> local_indices(num_dofs);

        for (int j = 0; j < num_dofs; ++j)
        {
            local_indices[j] = test.at(edge_dofs[j]);

            int row = col_marker[edge_dofs[j]];
            assert(row >= 0);

            for (int k = 0; k < num_dofs; ++k)
            {
                int col = col_marker[edge_dofs[k]];
                assert(col >= 0);

                sub(j, k) = M_fine_elem[vert](row, col);
            }
        }

        M_local.AddSubMatrix(local_indices, local_indices, sub);

        ClearMarker(col_marker, fine_dofs);
        if (!fine)
        {
            //using linalgcpp::operator<<;
            //std::cout << "Face dofs:" << edge_dofs;
            //std::cout << "local indices:" << local_indices;
            //submat.Print("Sub matrix:");
        }
        //*/

    }



    if (!fine && M_local.Rows() < 80)
    {
        //using linalgcpp::operator<<;
        //std::cout << "Face dofs:" << fine_dofs;
        //M_local.Print("M face:", std::cout, 6, 2);
    }
}

std::vector<DenseMatrix> GraphCoarsen::BuildElemM(const MixedMatrix& mgl,
                                                  const SparseMatrix& agg_cdof_edge) const
{
    int num_aggs = gt_.agg_ext_edge_.Rows();
    int num_faces = gt_.face_edge_local_.Rows();

    std::vector<DenseMatrix> M_el(num_aggs);

    DenseMatrix h_F_potential;
    DenseMatrix h_D_trace;
    DenseMatrix bub_block;
    DenseMatrix ortho_vects;
    DenseMatrix DTT_F;
    DenseMatrix DTT_B;

    for (int agg = 0; agg < num_aggs; ++agg)
    {
        linalgcpp::HStack(F_potential_[agg], h_F_potential);
        linalgcpp::HStack(D_trace_[agg], h_D_trace);

        int num_bubbles = vertex_targets_[agg].Cols() - 1;
        int num_traces = h_F_potential.Cols();

        M_el[agg].SetSize(num_bubbles + num_traces);
        M_el[agg] = 0.0;

        if (num_bubbles > 0)
        {
            int total_cols = vertex_targets_[agg].Cols();
            vertex_targets_[agg].GetCol(1, total_cols, ortho_vects);

            ortho_vects.MultAT(B_potential_[agg], bub_block);
            M_el[agg].SetSubMatrix(0, 0, bub_block);
        }

        h_D_trace.MultAT(h_F_potential, DTT_F);
        h_D_trace.MultAT(B_potential_[agg], DTT_B);

        M_el[agg].SetSubMatrix(num_bubbles, num_bubbles, DTT_F);
        M_el[agg].SetSubMatrix(num_bubbles, 0, DTT_B);
        M_el[agg].SetSubMatrixTranspose(0, num_bubbles, DTT_B);
    }

    DenseMatrix M_local;
    DenseMatrix edge_target_T;
    DenseMatrix edge_target_T_M;
    DenseMatrix trace_across;

    std::vector<int> agg_indices;

    //SparseMatrix edge_vertex = mgl.LocalD().Transpose();
    //SparseMatrix vertex_agg = gt_.agg_vertex_local_.Transpose();
    //SparseMatrix edof_vertex = mgl.vertex_edof.Transpose();
    //SparseMatrix edof_vdof = edof_vertex.Mult(mgl.vertex_vdof);
    //SparseMatrix vertex_agg = agg_vertexdof_.Transpose();
    //SparseMatrix edof_vertex = mgl.vertex_edof.Transpose();
    SparseMatrix edge_vertex = mgl.edge_edof.Mult(mgl.vertex_edof.Transpose());
    SparseMatrix vertex_agg = gt_.agg_vertex_local_.Transpose();

    for (int face = 0; face < num_faces; ++face)
    {
        std::vector<int> fine_edges = gt_.face_edge_local_.GetIndices(face);
        std::vector<int> fine_dofs = face_edgedof_.GetIndices(face);

        std::vector<int> coarse_dofs = face_cdof_.GetIndices(face);
        std::vector<int> aggs = gt_.face_agg_local_.GetIndices(face);
        agg_indices.resize(coarse_dofs.size());

        edge_targets_[face].Transpose(edge_target_T);

        for (auto agg : aggs)
        {
            //BuildAggFaceM(mgl, face, agg, vertex_agg, edge_vertex, col_marker_, M_local);

            ///*
            auto test = mgl.LocalM().GetSubMatrix(fine_dofs, fine_dofs, col_marker_);
            test /= 2.0;
            test.ToDense(M_local);
            //*/
            //if (!fine) M_local.Print("M submatrix:", std::cout, 8, 3);
            //if (!fine) edge_target_T.Print("Edge taret", std::cout, 8, 3);


            edge_target_T.Mult(M_local, edge_target_T_M);
            edge_target_T_M.Mult(edge_targets_[face], trace_across);

            std::vector<int> edges = agg_cdof_edge.GetIndices(agg);
            SetMarker(col_marker_, edges);

            int size = coarse_dofs.size();
            for (int i = 0; i < size; ++i)
            {
                agg_indices[i] = col_marker_[coarse_dofs[i]];
                assert(agg_indices[i] > -1);
            }

            //M_el[agg].Print("M El:");
            //trace_across.Print("Trace Across:");

            M_el[agg].AddSubMatrix(agg_indices, agg_indices, trace_across);
            //M_el[agg].Print("M El + trace:");

            ClearMarker(col_marker_, edges);
        }
    }

    return M_el;
}

DenseMatrix GraphCoarsen::RestrictLocal(const DenseMatrix& ext_mat,
                                        std::vector<int>& global_marker,
                                        const std::vector<int>& ext_indices,
                                        const std::vector<int>& local_indices) const
{
    SetMarker(global_marker, ext_indices);

    int local_size = local_indices.size();

    std::vector<int> row_map(local_size);

    for (int i = 0; i < local_size; ++i)
    {
        assert(global_marker[local_indices[i]] >= 0);
        row_map[i] = global_marker[local_indices[i]];
    }

    ClearMarker(global_marker, ext_indices);

    return ext_mat.GetRow(row_map);
}

std::vector<int> GraphCoarsen::GetExtDofs(const ParMatrix& mat_ext, int row) const
{
    const auto& diag = mat_ext.GetDiag();
    const auto& offd = mat_ext.GetOffd();

    auto diag_dofs = diag.GetIndices(row);
    auto offd_dofs = offd.GetIndices(row);

    int diag_size = diag.Cols();

    for (auto i : offd_dofs)
    {
        diag_dofs.push_back(i + diag_size);
    }

    return diag_dofs;
}

ParMatrix GraphCoarsen::MakeExtPermutation(const ParMatrix& parmat) const
{
    MPI_Comm comm = parmat.GetComm();

    const auto& diag = parmat.GetDiag();
    const auto& offd = parmat.GetOffd();
    const auto& colmap = parmat.GetColMap();

    int num_diag = diag.Cols();
    int num_offd = offd.Cols();
    int num_ext = num_diag + num_offd;

    const auto& mat_starts = parmat.GetColStarts();
    auto ext_starts = parlinalgcpp::GenerateOffsets(comm, num_ext);

    SparseMatrix perm_diag = SparseIdentity(num_ext, num_diag);
    SparseMatrix perm_offd = SparseIdentity(num_ext, num_offd, num_diag);

    return ParMatrix(comm, ext_starts, mat_starts, std::move(perm_diag), std::move(perm_offd), colmap);
}

MixedMatrix GraphCoarsen::Coarsen(const MixedMatrix& mgl) const
{
    SparseMatrix agg_cdof_edge = BuildAggCDofEdge();
    auto M_elem = BuildElemM(mgl, agg_cdof_edge);

    SparseMatrix D_c = BuildCoarseD();
    SparseMatrix W_c;

    if (mgl.LocalW().Rows() == P_vertex_.Rows())
    {
        SparseMatrix P_vertex_T = P_vertex_.Transpose();
        W_c = P_vertex_T.Mult(mgl.LocalW().Mult(P_vertex_));
    }

    ParMatrix edge_true_edge = BuildEdgeTrueEdge();

    MixedMatrix mm(std::move(M_elem), std::move(agg_cdof_edge),
                   std::move(D_c), std::move(W_c),
                   //std::move(edge_true_edge),
                   (edge_true_edge),
                   BuildAggCDofVertex(),
                   face_cdof_);

    mm.vertex_vdof = agg_vertexdof_.Mult(P_vertex_);
    mm.vertex_edof = agg_edgedof_.Mult(P_edge_);
    mm.vertex_bdof = agg_bubble_dof_;
    mm.edge_edof = face_edgedof_.Mult(P_edge_);

    mm.vertex_vdof = 1.0;
    mm.vertex_edof = 1.0;
    mm.edge_edof = 1.0;
    mm.constant_vect_ = P_vertex_.MultAT(mgl.constant_vect_);

    mm.AssembleM();

    //if (P_vertex_.Rows() < 80 && MyId() == 0)
    //if (P_vertex_.Rows() < 80 && MyId() == 0 && false)
    {
        SparseMatrix P_vertex_T = P_vertex_.Transpose();
        SparseMatrix P_edge_T = P_edge_.Transpose();

        SparseMatrix M_c = P_edge_T.Mult(mgl.LocalM()).Mult(P_edge_);
        SparseMatrix D_c = P_vertex_T.Mult(mgl.LocalD()).Mult(P_edge_);

        mm.M_local_ = M_c;
        mm.M_global_ = parlinalgcpp::RAP(ParMatrix(mgl.GlobalM().GetComm(), M_c), edge_true_edge);

        /*
        P_vertex_T.Mult(P_vertex_).PrintDense("PT P");
        P_edge_.PrintDense("P edge");
        P_vertex_.PrintDense("P vertex");

        int num_agg = gt_.agg_vertex_local_.Rows();

        for (int i = 0; i < num_agg; ++i)
        {
            mm.AssembleElem(i).PrintDense("Elem");
        }

        mm.LocalM().ToDense().Print("M c Assemble:", std::cout, 15, 10);
        M_c.ToDense().Print("M c RAP:", std::cout, 15, 10);
        D_c.ToDense().Print("D c RAP:", std::cout, 15, 10);
        mm.constant_vect_.Print("Coarse constant");

        auto m_diff = (M_c.ToDense() - mm.LocalM().ToDense());
        m_diff.Print("M c Diff:", std::cout, 15, 10);

        //auto m_tst = P_edge_.Mult(m_diff).Mult(P_edge_T.ToDense());
        //m_tst.Print("M mid Diff:", std::cout, 8, 4);

        (D_c.ToDense() - mm.LocalD().ToDense()).Print("D c Diff:", std::cout, 15, 10);
        */
    }

    fine = false;

    return mm;
}

Vector GraphCoarsen::Interpolate(const VectorView& coarse_vect) const
{
    return P_vertex_.Mult(coarse_vect);
}

void GraphCoarsen::Interpolate(const VectorView& coarse_vect, VectorView fine_vect) const
{
    P_vertex_.Mult(coarse_vect, fine_vect);
}

Vector GraphCoarsen::Restrict(const VectorView& fine_vect) const
{
    return P_vertex_.MultAT(fine_vect);
}

void GraphCoarsen::Restrict(const VectorView& fine_vect, VectorView coarse_vect) const
{
    P_vertex_.MultAT(fine_vect, coarse_vect);
}

BlockVector GraphCoarsen::Interpolate(const BlockVector& coarse_vect) const
{
    std::vector<int> fine_offsets = {0, P_edge_.Rows(), P_edge_.Rows() + P_vertex_.Rows()};
    BlockVector fine_vect(fine_offsets);

    Interpolate(coarse_vect, fine_vect);

    return fine_vect;
}

void GraphCoarsen::Interpolate(const BlockVector& coarse_vect, BlockVector& fine_vect) const
{
    P_edge_.Mult(coarse_vect.GetBlock(0), fine_vect.GetBlock(0));
    P_vertex_.Mult(coarse_vect.GetBlock(1), fine_vect.GetBlock(1));
}

BlockVector GraphCoarsen::Restrict(const BlockVector& fine_vect) const
{
    std::vector<int> coarse_offsets = {0, P_edge_.Cols(), P_edge_.Cols() + P_vertex_.Cols()};
    BlockVector coarse_vect(coarse_offsets);

    Restrict(fine_vect, coarse_vect);

    return coarse_vect;
}

void GraphCoarsen::Restrict(const BlockVector& fine_vect, BlockVector& coarse_vect) const
{
    P_edge_.MultAT(fine_vect.GetBlock(0), coarse_vect.GetBlock(0));
    P_vertex_.MultAT(fine_vect.GetBlock(1), coarse_vect.GetBlock(1));
}


} // namespace smoothg
