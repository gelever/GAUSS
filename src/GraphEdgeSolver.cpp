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

#include "GraphEdgeSolver.hpp"

#include "LocalEigenSolver.hpp"

namespace smoothg
{

GraphEdgeSolver::GraphEdgeSolver(const SparseMatrix& M, const SparseMatrix& D)
      : rhs_({0, M.Rows(), M.Rows() + D.Rows()}),
        sol_({0, M.Rows(), M.Rows() + D.Rows()})
{
    rhs_ = 0.0;

    linalgcpp::BlockMatrix<double> block({0, M.Rows(), M.Rows() + D.Rows()});

    CooMatrix W_coo(D.Rows(), D.Rows());
    W_coo.Add(0, 0, 1);

    SparseMatrix W = W_coo.ToSparse();

    SparseMatrix D_elim(D);
    D_elim.EliminateRow(0);
    SparseMatrix DT_elim = D_elim.Transpose();

    block.SetBlock(0, 0, M);
    block.SetBlock(0, 1, DT_elim);
    block.SetBlock(1, 0, D_elim);
    block.SetBlock(1, 1, W);

    block_Ainv_ = SparseSolver(block.Combine());

}

GraphEdgeSolver::GraphEdgeSolver(const GraphEdgeSolver& other) noexcept
    : block_Ainv_(other.block_Ainv_),
      rhs_(other.rhs_),
      sol_(other.sol_)
{

}

GraphEdgeSolver::GraphEdgeSolver(GraphEdgeSolver&& other) noexcept
{
    swap(*this, other);
}

GraphEdgeSolver& GraphEdgeSolver::operator=(GraphEdgeSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(GraphEdgeSolver& lhs, GraphEdgeSolver& rhs) noexcept
{
    swap(lhs.block_Ainv_, rhs.block_Ainv_);
    swap(lhs.rhs_, rhs.rhs_);
    swap(lhs.sol_, rhs.sol_);
}

Vector GraphEdgeSolver::Mult(const VectorView& input) const
{
    Vector vect(rhs_.GetBlock(0).size());

    Mult(input, vect);

    return vect;
}

void GraphEdgeSolver::Mult(const VectorView& input, VectorView output) const
{
    rhs_.GetBlock(1) = input;
    rhs_.GetBlock(1)[0] = 0.0;

    block_Ainv_.Mult(rhs_, sol_);

    output = sol_.GetBlock(0);
}

void GraphEdgeSolver::Mult(const VectorView& input, VectorView sigma_sol, VectorView u_sol) const
{
    rhs_.GetBlock(1) = input;
    rhs_.GetBlock(1)[0] = 0.0;

    block_Ainv_.Mult(rhs_, sol_);

    sigma_sol = sol_.GetBlock(0);

    u_sol = sol_.GetBlock(1);
    u_sol *= -1.0;
    //SubAvg(u_sol);
}

DenseMatrix GraphEdgeSolver::Mult(const DenseMatrix& input) const
{
    DenseMatrix output;
    Mult(input, output);

    return output;
}

void GraphEdgeSolver::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    int rows = rhs_.GetBlock(0).size();
    int cols = input.Cols();

    output.SetSize(rows, cols);

    for (int i = 0; i < cols; ++i)
    {
        const VectorView& in_col = input.GetColView(i);
        VectorView out_col = output.GetColView(i);

        Mult(in_col, out_col);
    }
}

void GraphEdgeSolver::Mult(const DenseMatrix& input, DenseMatrix& sigma_sol,
                           DenseMatrix& u_sol) const
{
    int rows = rhs_.GetBlock(0).size();
    int cols = input.Cols();

    sigma_sol.SetSize(rows, cols);
    u_sol.SetSize(rhs_.GetBlock(1).size(), cols);

    for (int i = 0; i < cols; ++i)
    {
        const VectorView& in_col = input.GetColView(i);
        VectorView sigma_col = sigma_sol.GetColView(i);
        VectorView u_col = u_sol.GetColView(i);

        Mult(in_col, sigma_col, u_col);
    }
}

void GraphEdgeSolver::OffsetMult(int offset, const DenseMatrix& input, DenseMatrix& output) const
{
    OffsetMult(offset, input.Cols(), input, output);
}

void GraphEdgeSolver::OffsetMult(int offset, const DenseMatrix& input, DenseMatrix& sigma_sol,
                                 DenseMatrix& u_sol) const
{
    OffsetMult(offset, input.Cols(), input, sigma_sol, u_sol);
}

void GraphEdgeSolver::OffsetMult(int start, int end, const DenseMatrix& input,
                                 DenseMatrix& output) const
{
    assert(start >= 0);
    assert(end <= input.Cols());

    int size = end - start;

    output.SetSize(rhs_.GetBlock(0).size(), size);

    for (int i = 0; i < size; ++i)
    {
        const VectorView& in_col = input.GetColView(i + start);
        VectorView out_col = output.GetColView(i);

        Mult(in_col, out_col);
    }
}

void GraphEdgeSolver::OffsetMult(int start, int end, const DenseMatrix& input,
                                 DenseMatrix& sigma_sol, DenseMatrix& u_sol) const
{
    assert(start >= 0);
    assert(end <= input.Cols());

    int size = end - start;

    sigma_sol.SetSize(rhs_.GetBlock(0).size(), size);
    u_sol.SetSize(rhs_.GetBlock(1).size(), size);

    for (int i = 0; i < size; ++i)
    {
        const VectorView& in_col = input.GetColView(i + start);
        VectorView sigma_col = sigma_sol.GetColView(i);
        VectorView u_col = u_sol.GetColView(i);

        Mult(in_col, sigma_col, u_col);
    }
}

} // namespace smoothg
