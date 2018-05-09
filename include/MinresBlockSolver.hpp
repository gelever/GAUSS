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

/**
   @file MinresBlockSolver.hpp

   @brief Given a graph in mixed form, solve the resulting system with
   preconditioned MINRES
*/

#ifndef __MINRESBLOCKSOLVER_HPP
#define __MINRESBLOCKSOLVER_HPP

#include <memory>
#include <assert.h>

#include "Utilities.hpp"
#include "MixedMatrix.hpp"
#include "MGLSolver.hpp"

namespace smoothg
{

/**
   @brief Block diagonal preconditioned MINRES solver for saddle point
   problem.

   Given matrix M and D, setup and solve the graph Laplacian problem
   \f[
     \left( \begin{array}{cc}
       M&  D^T \\
       D&  -W
     \end{array} \right)
     \left( \begin{array}{c}
       u \\ p
     \end{array} \right)
     =
     \left( \begin{array}{c}
       f \\ g
     \end{array} \right)
   \f]
   using MinRes with a block-diagonal preconditioner.

   This class and its implementation owes a lot to MFEM example ex5p
*/
class MinresBlockSolver : public MGLSolver
{
public:
    /** @brief Default Constructor */
    MinresBlockSolver() = default;

    /** @brief Constructor from a mixed matrix
        @param mgl mixed matrix information
    */
    template <typename T>
    MinresBlockSolver(const MixedMatrix<T>& mgl);

    /** @brief Copy Constructor */
    MinresBlockSolver(const MinresBlockSolver& other) noexcept;

    /** @brief Move Constructor */
    MinresBlockSolver(MinresBlockSolver&& other) noexcept;

    /** @brief Assignment Operator */
    MinresBlockSolver& operator=(MinresBlockSolver other) noexcept;

    /** @brief Swap two solvers */
    friend void swap(MinresBlockSolver& lhs, MinresBlockSolver& rhs) noexcept;

    /** @brief Default Destructor */
    ~MinresBlockSolver() noexcept = default;

    /** @brief Use block-preconditioned MINRES to solve the problem.
        @param rhs Right hand side
        @param sol Solution
    */
    void Solve(const BlockVector& rhs, BlockVector& sol) const override;

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

protected:

    ParMatrix M_;
    ParMatrix D_;
    ParMatrix DT_;
    ParMatrix W_;

    ParMatrix edge_true_edge_;

    MPI_Comm comm_;
    int myid_;

private:
    linalgcpp::BlockOperator op_;
    linalgcpp::BlockOperator prec_;

    parlinalgcpp::ParDiagScale M_prec_;
    parlinalgcpp::BoomerAMG schur_prec_;

    linalgcpp::PMINRESSolver pminres_;

    mutable BlockVector true_rhs_;
    mutable BlockVector true_sol_;

    bool use_w_;
};

template <typename T>
MinresBlockSolver::MinresBlockSolver(const MixedMatrix<T>& mgl)
    : MGLSolver(mgl.Offsets()), M_(mgl.GlobalM()), D_(mgl.GlobalD()), W_(mgl.GlobalW()),
      edge_true_edge_(mgl.EdgeTrueEdge()),
      comm_(M_.GetComm()), myid_(M_.GetMyId()),
      op_(mgl.TrueOffsets()), prec_(mgl.TrueOffsets()),
      true_rhs_(mgl.TrueOffsets()), true_sol_(mgl.TrueOffsets()),
      use_w_(mgl.CheckW())
{
    if (!use_w_ && myid_ == 0)
    {
        D_.EliminateRow(0);
    }

    DT_ = D_.Transpose();

    SparseMatrix M_diag(M_.GetDiag().GetDiag());
    ParMatrix MinvDT = DT_;
    MinvDT.InverseScaleRows(M_diag);
    ParMatrix schur_block = D_.Mult(MinvDT);

    if (!use_w_)
    {
        CooMatrix elim_dof(D_.Rows(), D_.Rows());

        if (myid_ == 0)
        {
            elim_dof.Add(0, 0, 1.0);
        }

        SparseMatrix W = elim_dof.ToSparse();
        W_ = ParMatrix(D_.GetComm(), D_.GetRowStarts(), std::move(W));
    }
    else
    {
        schur_block = parlinalgcpp::ParSub(schur_block, W_);
    }

    M_prec_ = parlinalgcpp::ParDiagScale(M_);
    schur_prec_ = parlinalgcpp::BoomerAMG(std::move(schur_block));

    op_.SetBlock(0, 0, M_);
    op_.SetBlock(0, 1, DT_);
    op_.SetBlock(1, 0, D_);
    op_.SetBlock(1, 1, W_);

    prec_.SetBlock(0, 0, M_prec_);
    prec_.SetBlock(1, 1, schur_prec_);

    pminres_ = linalgcpp::PMINRESSolver(op_, prec_, max_num_iter_, rtol_,
                                        atol_, 0, parlinalgcpp::ParMult);

    if (myid_ == 0)
    {
        SetPrintLevel(print_level_);
    }

    nnz_ = M_.nnz() + DT_.nnz() + D_.nnz() + W_.nnz();
}


} // namespace smoothg

#endif
