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

    @brief Contains MixedMatrix class, which encapsulates a mixed form of graph.
 */

#ifndef __MIXEDMATRIX_HPP__
#define __MIXEDMATRIX_HPP__

#include "Utilities.hpp"
#include "Graph.hpp"

namespace smoothg
{

/**
   @brief Encapuslates the mixed form of a graph in saddle-point form.

   The given data is a vertex_edge table and weights in some form.

   This is essentially a container for a weight matrix and a D matrix.

   Two types of element matrices are supported:
   vector for when M is diagonal and dense matrix otherwise.
*/

template <typename T>
class MixedMatrix
{
public:
    /** @brief Default Constructor */
    MixedMatrix() = default;

    /** @brief Generates local matrices given global graph information
        @param graph Global graph information
        @param global_weight Global edge weights
        @param W_global optional global W block
    */
    MixedMatrix(const Graph& graph);

    /** @brief Constructor with given local matrices
        @param M_elem Local M element matrices
        @param elem_dof element to dof relationship
        @param D_local Local D
        @param W_local Local W
        @param edge_true_edge Edge to true edge relationship
    */
    MixedMatrix(std::vector<T> M_elem, SparseMatrix elem_dof,
                    SparseMatrix D_local, SparseMatrix W_local,
                    ParMatrix edge_true_edge);

    /** @brief Default Destructor */
    virtual ~MixedMatrix() noexcept = default;

    /** @brief Copy Constructor */
    MixedMatrix(const MixedMatrix& other) noexcept;

    /** @brief Move Constructor */
    MixedMatrix(MixedMatrix&& other) noexcept;

    /** @brief Assignment Operator */
    MixedMatrix& operator=(MixedMatrix other) noexcept;

    /** @brief Swap two mixed matrices */
    template <typename U>
    friend void swap(MixedMatrix<U>& lhs, MixedMatrix<U>& rhs) noexcept;

    /** @brief Assemble M from element matrices */
    void AssembleM();

    /** @brief Assemble scaled M from element matrices
        @param agg_weight weights per aggregate
    */
    void AssembleM(const std::vector<double>& agg_weight);

    /** @brief Access element matrices */
    const std::vector<T>& GetElemM() const { return M_elem_; }

    /** @brief Access element to dof relationship */
    const SparseMatrix& GetElemDof() const { return elem_dof_; }

    /* @brief Local size of mixed matrix, number of edges + number of vertices */
    int Rows() const;

    /* @brief Local size of mixed matrix, number of edges + number of vertices */
    int Cols() const;

    /* @brief Global size of mixed matrix, number of edges + number of vertices */
    int GlobalRows() const;

    /* @brief Global size of mixed matrix, number of edges + number of vertices */
    int GlobalCols() const;

    /* @brief Local number of nun zero entries */
    int NNZ() const;

    /* @brief Global number of nun zero entries */
    int GlobalNNZ() const;

    /* @brief Check if the W block is non empty
       @returns True if W is non empty
    */
    bool CheckW() const;

    /* @brief Computes the global primal form of the mixed matrix: A = DM^{-1}D^T
       @warning Requires that M is diagonal since it will be inverted
    */
    ParMatrix ToPrimal() const;

    /* @brief Get Local M */
    const SparseMatrix& LocalM() const { return M_local_; }

    /* @brief Get Local D  */
    const SparseMatrix& LocalD() const { return D_local_; }

    /* @brief Get Local W  */
    const SparseMatrix& LocalW() const { return W_local_; }

    /* @brief Get Global M  */
    const ParMatrix& GlobalM() const { return M_global_; }

    /* @brief Get Global D  */
    const ParMatrix& GlobalD() const { return D_global_; }

    /* @brief Get Global W  */
    const ParMatrix& GlobalW() const { return W_global_; }

    /* @brief Get Edge True Edge */
    const ParMatrix& EdgeTrueEdge() const { return edge_true_edge_; }

    /* @brief Block offsets */
    const std::vector<int>& Offsets() const { return offsets_; }

    /* @brief Block true offsets */
    const std::vector<int>& TrueOffsets() const { return true_offsets_; }

    // TEMP: stuff from coarsener
    SparseMatrix agg_vertexdof_;
    SparseMatrix agg_edgedof_;
    int num_multiplier_dofs_;

protected:
    void Init();

    SparseMatrix MakeLocalD(const ParMatrix& edge_true_edge,
                            const SparseMatrix& vertex_edge) const;



    ParMatrix edge_true_edge_;

    // Local blocks
    SparseMatrix M_local_;
    SparseMatrix D_local_;
    SparseMatrix W_local_;

    // Global blocks
    ParMatrix M_global_;
    ParMatrix D_global_;
    ParMatrix W_global_;

    std::vector<int> offsets_;
    std::vector<int> true_offsets_;

    // Element information
    std::vector<T> M_elem_;
    SparseMatrix elem_dof_;
};

using VectorMixedMatrix = MixedMatrix<std::vector<double>>;
using DenseMixedMatrix = MixedMatrix<DenseMatrix>;

template <typename T>
MixedMatrix<T>::MixedMatrix(std::vector<T> M_elem, SparseMatrix elem_dof,
                                    SparseMatrix D_local, SparseMatrix W_local,
                                    ParMatrix edge_true_edge)
    : edge_true_edge_(std::move(edge_true_edge)),
      D_local_(std::move(D_local)),
      W_local_(std::move(W_local)),
      M_elem_(std::move(M_elem)),
      elem_dof_(std::move(elem_dof))
{
    Init();
}


template <typename T>
void MixedMatrix<T>::Init()
{
    MPI_Comm comm = edge_true_edge_.GetComm();

    auto starts = parlinalgcpp::GenerateOffsets(comm, {D_local_.Rows(), D_local_.Cols()});
    std::vector<HYPRE_Int>& vertex_starts = starts[0];
    std::vector<HYPRE_Int>& edge_starts = starts[1];

    ParMatrix D_d(comm, vertex_starts, edge_starts, D_local_);
    D_global_ = D_d.Mult(edge_true_edge_);

    if (M_local_.Rows() == D_local_.Cols())
    {
        ParMatrix M_d(comm, edge_starts, M_local_);
        M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
    }

    if (W_local_.Rows() == D_local_.Rows())
    {
        W_global_ = ParMatrix(comm, vertex_starts, W_local_);
    }

    offsets_ = {0, D_local_.Cols(), D_local_.Cols() + D_local_.Rows()};
    true_offsets_ = {0, D_global_.Cols(), D_global_.Cols() + D_global_.Rows()};
}

template <typename T>
MixedMatrix<T>::MixedMatrix(const MixedMatrix<T>& other) noexcept
    : edge_true_edge_(other.edge_true_edge_),
      M_local_(other.M_local_),
      D_local_(other.D_local_),
      W_local_(other.W_local_),
      M_global_(other.M_global_),
      D_global_(other.D_global_),
      W_global_(other.W_global_),
      offsets_(other.offsets_),
      true_offsets_(other.true_offsets_),
      M_elem_(other.M_elem_),
      elem_dof_(other.elem_dof_),
      agg_vertexdof_(other.agg_vertexdof_),
      agg_edgedof_(other.agg_edgedof_),
      num_multiplier_dofs_(other.num_multiplier_dofs_)
{

}

template <typename T>
MixedMatrix<T>::MixedMatrix(MixedMatrix<T>&& other) noexcept
{
    swap(*this, other);
}

template <typename T>
MixedMatrix<T>& MixedMatrix<T>::operator=(MixedMatrix<T> other) noexcept
{
    swap(*this, other);

    return *this;
}

template <typename T>
void swap(MixedMatrix<T>& lhs, MixedMatrix<T>& rhs) noexcept
{
    swap(lhs.edge_true_edge_, rhs.edge_true_edge_);

    swap(lhs.M_local_, rhs.M_local_);
    swap(lhs.D_local_, rhs.D_local_);
    swap(lhs.W_local_, rhs.W_local_);

    swap(lhs.M_global_, rhs.M_global_);
    swap(lhs.D_global_, rhs.D_global_);
    swap(lhs.W_global_, rhs.W_global_);

    std::swap(lhs.offsets_, rhs.offsets_);
    std::swap(lhs.true_offsets_, rhs.true_offsets_);

    swap(lhs.M_elem_, rhs.M_elem_);
    swap(lhs.elem_dof_, rhs.elem_dof_);

    swap(lhs.agg_vertexdof_, rhs.agg_vertexdof_);
    swap(lhs.agg_edgedof_, rhs.agg_edgedof_);
    std::swap(lhs.num_multiplier_dofs_, rhs.num_multiplier_dofs_);
}

template <typename T>
int MixedMatrix<T>::Rows() const
{
    return D_local_.Rows() + D_local_.Cols();
}

template <typename T>
int MixedMatrix<T>::Cols() const
{
    return D_local_.Rows() + D_local_.Cols();
}

template <typename T>
int MixedMatrix<T>::GlobalRows() const
{
    return D_global_.GlobalRows() + D_global_.GlobalCols();
}

template <typename T>
int MixedMatrix<T>::GlobalCols() const
{
    return D_global_.GlobalRows() + D_global_.GlobalCols();
}

template <typename T>
int MixedMatrix<T>::NNZ() const
{
    return M_local_.nnz() + (2 * D_local_.nnz())
           + W_local_.nnz();
}

template <typename T>
int MixedMatrix<T>::GlobalNNZ() const
{
    return M_global_.nnz() + (2 * D_global_.nnz())
           + W_global_.nnz();
}

template <typename T>
bool MixedMatrix<T>::CheckW() const
{
    int local_size = W_global_.Rows();
    int global_size;

    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_MAX, D_global_.GetComm());

    const double zero_tol = 1e-6;

    return global_size > 0 && W_global_.MaxNorm() > zero_tol;
}

template <typename T>
void MixedMatrix<T>::AssembleM()
{
    int M_size = D_local_.Cols();
    CooMatrix M_coo(M_size, M_size);

    int num_aggs = M_elem_.size();

    for (int i = 0; i < num_aggs; ++i)
    {
        std::vector<int> dofs = elem_dof_.GetIndices(i);

        M_coo.Add(dofs, dofs, M_elem_[i]);
    }

    M_local_ = M_coo.ToSparse();
    ParMatrix M_d(edge_true_edge_.GetComm(), edge_true_edge_.GetRowStarts(), M_local_);
    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
}

template <typename T>
void MixedMatrix<T>::AssembleM(const std::vector<double>& agg_weight)
{
    assert(agg_weight.size() == M_elem_.size());

    int M_size = D_local_.Cols();
    CooMatrix M_coo(M_size, M_size);

    int num_aggs = M_elem_.size();

    for (int i = 0; i < num_aggs; ++i)
    {
        double scale = 1.0 / agg_weight[i];
        std::vector<int> dofs = elem_dof_.GetIndices(i);

        M_coo.Add(dofs, dofs, scale, M_elem_[i]);
    }

    M_local_ = M_coo.ToSparse();
    ParMatrix M_d(edge_true_edge_.GetComm(), edge_true_edge_.GetRowStarts(), M_local_);
    M_global_ = parlinalgcpp::RAP(M_d, edge_true_edge_);
}

template <typename T>
SparseMatrix MixedMatrix<T>::MakeLocalD(const ParMatrix& edge_true_edge,
                                     const SparseMatrix& vertex_edge) const
{
    SparseMatrix edge_vertex = vertex_edge.Transpose();

    std::vector<int> indptr = edge_vertex.GetIndptr();
    std::vector<int> indices = edge_vertex.GetIndices();
    std::vector<double> data = edge_vertex.GetData();

    int num_edges = edge_vertex.Rows();
    int num_vertices = edge_vertex.Cols();

    const SparseMatrix& owned_edges = edge_true_edge.GetDiag();

    for (int i = 0; i < num_edges; i++)
    {
        const int row_edges = edge_vertex.RowSize(i);
        assert(row_edges == 1 || row_edges == 2);

        data[indptr[i]] = 1.;

        if (row_edges == 2)
        {
            data[indptr[i] + 1] = -1.;
        }
        else if (owned_edges.RowSize(i) == 0)
        {
            assert(row_edges == 1);
            data[indptr[i]] = -1.;
        }
    }

    SparseMatrix DT(std::move(indptr), std::move(indices), std::move(data),
                    num_edges, num_vertices);

    return DT.Transpose();
}

} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
