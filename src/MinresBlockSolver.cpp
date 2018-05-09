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
   @file

   @brief Implements MinresBlockSolver object.
*/

#include "MinresBlockSolver.hpp"

namespace smoothg
{

MinresBlockSolver::MinresBlockSolver(const MinresBlockSolver& other) noexcept
    : MGLSolver(other), op_(other.op_), prec_(other.prec_),
      M_prec_(other.M_prec_), schur_prec_(other.schur_prec_),
      pminres_(other.pminres_),
      true_rhs_(other.true_rhs_), true_sol_(other.true_sol_)
{

}

MinresBlockSolver::MinresBlockSolver(MinresBlockSolver&& other) noexcept
{
    swap(*this, other);
}

MinresBlockSolver& MinresBlockSolver::operator=(MinresBlockSolver other) noexcept
{
    swap(*this, other);

    return *this;
}

void swap(MinresBlockSolver& lhs, MinresBlockSolver& rhs) noexcept
{
    swap(static_cast<MGLSolver&>(lhs),
         static_cast<MGLSolver&>(rhs));

    swap(lhs.op_, rhs.op_);
    swap(lhs.prec_, rhs.prec_);
    swap(lhs.M_prec_, rhs.M_prec_);
    swap(lhs.schur_prec_, rhs.schur_prec_);
    swap(lhs.pminres_, rhs.pminres_);
    swap(lhs.true_rhs_, rhs.true_rhs_);
    swap(lhs.true_sol_, rhs.true_sol_);
}

void MinresBlockSolver::Solve(const BlockVector& rhs, BlockVector& sol) const
{
    Timer timer(Timer::Start::True);

    edge_true_edge_.MultAT(rhs.GetBlock(0), true_rhs_.GetBlock(0));
    true_rhs_.GetBlock(1) = rhs.GetBlock(1);
    true_sol_ = 0.0;

    if (!use_w_ && myid_ == 0)
    {
        true_rhs_[0] = 0.0;
    }

    pminres_.Mult(true_rhs_, true_sol_);
    num_iterations_ += pminres_.GetNumIterations();

    edge_true_edge_.Mult(true_sol_.GetBlock(0), sol.GetBlock(0));
    sol.GetBlock(1) = true_sol_.GetBlock(1);

    timer.Click();
    timing_ += timer.TotalTime();
}

void MinresBlockSolver::SetPrintLevel(int print_level)
{
    MGLSolver::SetPrintLevel(print_level);

    if (myid_ == 0)
    {
        pminres_.SetVerbose(print_level_);
    }
}

void MinresBlockSolver::SetMaxIter(int max_num_iter)
{
    MGLSolver::SetMaxIter(max_num_iter);

    pminres_.SetMaxIter(max_num_iter);
}

void MinresBlockSolver::SetRelTol(double rtol)
{
    MGLSolver::SetRelTol(rtol);

    pminres_.SetRelTol(rtol);
}

void MinresBlockSolver::SetAbsTol(double atol)
{
    MGLSolver::SetAbsTol(atol);

    pminres_.SetAbsTol(atol);
}

} // namespace smoothg

