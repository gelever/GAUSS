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

namespace smoothg
{

GraphEdgeSolver::GraphEdgeSolver(SparseMatrix M, SparseMatrix D)
    : is_diag_(IsDiag(M))
{
    assert(M.Rows() > 0);
    assert(D.Rows() > 0);

    if (is_diag_)
    {
        rhs_ = BlockVector({0, 0, D.Rows()});
        sol_ = BlockVector({0, 0, D.Rows()});

        MinvDT_ = D.Transpose();
        MinvDT_.InverseScaleRows(M.GetData());

        SparseMatrix A = D.Mult(MinvDT_);
        A.EliminateRowCol(0);

        block_Ainv_ = SparseSolver(std::move(A));
    }
    else
    {
        rhs_ = BlockVector({0, M.Rows(), M.Rows() + D.Rows()});
        sol_ = BlockVector({0, M.Rows(), M.Rows() + D.Rows()});

        linalgcpp::BlockMatrix<double> block({0, M.Rows(), M.Rows() + D.Rows()});

        int elim_dof = 0;

        CooMatrix W_coo(D.Rows(), D.Rows());
        W_coo.Add(elim_dof, elim_dof, 1);

        SparseMatrix W = W_coo.ToSparse();

        D.EliminateRow(elim_dof);
        SparseMatrix DT = D.Transpose();

        block.SetBlock(0, 0, std::move(M));
        block.SetBlock(0, 1, std::move(DT));
        block.SetBlock(1, 0, std::move(D));
        block.SetBlock(1, 1, std::move(W));

        block_Ainv_ = SparseSolver(block.Combine());
    }

    rhs_ = 0.0;
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
    int size = is_diag_ ? MinvDT_.Rows() : rhs_.GetBlock(0).size();

    Vector vect(size);

    Mult(input, vect);

    return vect;
}

void GraphEdgeSolver::Mult(const VectorView& input, VectorView output) const
{
    rhs_.GetBlock(1) = input;
    rhs_.GetBlock(1)[0] = 0.0;

    block_Ainv_.Mult(rhs_, sol_);

    if (is_diag_)
    {
        MinvDT_.Mult(sol_.GetBlock(1), output);
    }
    else
    {
        output = sol_.GetBlock(0);
    }
}

void GraphEdgeSolver::Mult(const VectorView& input, VectorView sigma_sol, VectorView u_sol) const
{
    rhs_.GetBlock(1) = input;
    rhs_.GetBlock(1)[0] = 0.0;

    block_Ainv_.Mult(rhs_, sol_);

    if (is_diag_)
    {
        MinvDT_.Mult(sol_.GetBlock(1), sigma_sol);
    }
    else
    {
        sol_.GetBlock(1) *= -1.0;
        sigma_sol = sol_.GetBlock(0);
    }

    u_sol = sol_.GetBlock(1);
}

DenseMatrix GraphEdgeSolver::Mult(const DenseMatrix& input) const
{
    DenseMatrix output;
    Mult(input, output);

    return output;
}

void GraphEdgeSolver::Mult(const DenseMatrix& input, DenseMatrix& output) const
{
    int rows = is_diag_ ? MinvDT_.Rows() : rhs_.GetBlock(0).size();
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
    int rows = is_diag_ ? MinvDT_.Rows() : rhs_.GetBlock(0).size();
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

    int rows = is_diag_ ? MinvDT_.Rows() : rhs_.GetBlock(0).size();
    int cols = end - start;

    output.SetSize(rows, cols);

    for (int i = 0; i < cols; ++i)
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

    int edges = is_diag_ ? MinvDT_.Rows() : rhs_.GetBlock(0).size();
    int cols = end - start;

    sigma_sol.SetSize(edges, cols);
    u_sol.SetSize(rhs_.GetBlock(1).size(), cols);

    for (int i = 0; i < cols; ++i)
    {
        const VectorView& in_col = input.GetColView(i + start);
        VectorView sigma_col = sigma_sol.GetColView(i);
        VectorView u_col = u_sol.GetColView(i);

        Mult(in_col, sigma_col, u_col);
    }
}

} // namespace smoothg
