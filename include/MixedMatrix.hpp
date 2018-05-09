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

class BaseElem
{
    public:
        virtual ~BaseElem() = default;

        virtual void Invert(DenseMatrix& inverse) const = 0;

        /// I dont like AddToCoo, but idk how to get derived data w/ casting
        /// At least it works for now
        virtual void AddToCoo(CooMatrix& coo, const std::vector<int>& indices,
                              double scale = 1.0) const = 0;
};

template <typename T>
class Elem : public BaseElem
{
    public:
        Elem(T elem) : data_(std::move(elem)) {}
        virtual ~Elem() = default;

        virtual void Invert(DenseMatrix& inverse) const override;
        virtual void AddToCoo(CooMatrix& coo, const std::vector<int>& indices,
                              double scale = 1.0) const override;
        const T& GetData() const { return data_; }

    private:
        T data_;
};

template<>
inline
void Elem<std::vector<double>>::Invert(DenseMatrix& inverse) const
{
    int size = data_.size();

    inverse.SetSize(size);
    inverse = 0.0;

    for (int i = 0; i < size; ++i)
    {
        inverse(i, i) = 1.0 / data_[i];
    }
}

template<>
inline
void Elem<DenseMatrix>::Invert(DenseMatrix& inverse) const
{
    int size = data_.Rows();
    inverse.SetSize(size);

    data_.Invert(inverse);
}

template <typename T>
void Elem<T>::AddToCoo(CooMatrix& coo, const std::vector<int>& indices, double scale) const
{
    coo.Add(indices, indices, scale, data_);
}

/**
   @brief Encapuslates the mixed form of a graph in saddle-point form.

   The given data is a vertex_edge table and weights in some form.

   This is essentially a container for a weight matrix and a D matrix.

   Two types of element matrices are supported:
   vector for when M is diagonal and dense matrix otherwise.
*/

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
    template <typename T>
    MixedMatrix(std::vector<T> M_elem, SparseMatrix elem_dof,
                    SparseMatrix D_local, SparseMatrix W_local,
                    ParMatrix edge_true_edge);

    /** @brief Default Destructor */
    virtual ~MixedMatrix() noexcept = default;

    /** @brief Copy Constructor */
    MixedMatrix(const MixedMatrix& other) noexcept = delete;

    /** @brief Move Constructor */
    MixedMatrix(MixedMatrix&& other) noexcept;

    /** @brief Assignment Operator */
    MixedMatrix& operator=(MixedMatrix&& other) noexcept;

    /** @brief Swap two mixed matrices */
    friend void swap(MixedMatrix& lhs, MixedMatrix& rhs) noexcept;

    /** @brief Assemble M from element matrices */
    void AssembleM();

    /** @brief Assemble scaled M from element matrices
        @param agg_weight weights per aggregate
    */
    void AssembleM(const std::vector<double>& agg_weight);

    /** @brief Access element matrices */
    const std::vector<std::unique_ptr<BaseElem>>& GetElemM() const { return M_elem_; }

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
    std::vector<std::unique_ptr<BaseElem>> M_elem_;
    SparseMatrix elem_dof_;
};

template <typename T>
MixedMatrix::MixedMatrix(std::vector<T> M_elem, SparseMatrix elem_dof,
                         SparseMatrix D_local, SparseMatrix W_local,
                         ParMatrix edge_true_edge)
    : edge_true_edge_(std::move(edge_true_edge)),
      D_local_(std::move(D_local)),
      W_local_(std::move(W_local)),
      elem_dof_(std::move(elem_dof))
{
    int num_elem = M_elem.size();

    M_elem_.resize(num_elem);

    for (int i = 0; i < num_elem; ++i)
    {
        M_elem_[i] = make_unique< Elem <T> >(std::move(M_elem[i]));
    }

    Init();
}


} // namespace smoothg

#endif /* __MIXEDMATRIX_HPP__ */
