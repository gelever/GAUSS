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

   @brief Implements HybridSolver object
*/

#include "HybridSolver.hpp"

namespace smoothg
{

void HybridSolver::InitSolver(SparseMatrix local_hybrid)
{
    if (myid_ == 0 && !use_w_)
    {
        local_hybrid.EliminateRowCol(0);
    }

    ParMatrix hybrid_d = ParMatrix(comm_, std::move(local_hybrid));

    pHybridSystem_ = parlinalgcpp::RAP(hybrid_d, multiplier_d_td_);
    nnz_ = pHybridSystem_.nnz();

    cg_ = linalgcpp::PCGSolver(pHybridSystem_, max_num_iter_, rtol_,
                               atol_, 0, parlinalgcpp::ParMult);
    if (myid_ == 0)
    {
        SetPrintLevel(print_level_);
    }

    // HypreBoomerAMG is broken if local size is zero
    int local_size = pHybridSystem_.Rows();
    int min_size;
    MPI_Allreduce(&local_size, &min_size, 1, MPI_INT, MPI_MIN, comm_);

    const bool use_prec = min_size > 0;
    if (use_prec)
    {
        prec_ = parlinalgcpp::BoomerAMG(pHybridSystem_);
        cg_.SetPreconditioner(prec_);
    }
    else
    {
        if (myid_ == 0)
        {
            // TODO(gelever1): create SMOOTHG_Warn or something of the sort
            // to make warnings optional
            printf("Warning: Not using preconditioner for Hybrid Solver!\n");
        }
    }

    trueHrhs_.SetSize(multiplier_d_td_.Cols());
    trueMu_.SetSize(trueHrhs_.size());
    Hrhs_.SetSize(num_multiplier_dofs_);
    Mu_.SetSize(num_multiplier_dofs_);
}

SparseMatrix HybridSolver::MakeEdgeDofMultiplier() const
{
    std::vector<int> indptr(num_edge_dofs_ + 1);
    std::iota(std::begin(indptr), std::begin(indptr) + num_multiplier_dofs_ + 1, 0);
    std::fill(std::begin(indptr) + num_multiplier_dofs_ + 1, std::end(indptr),
              indptr[num_multiplier_dofs_]);

    std::vector<int> indices(num_multiplier_dofs_);
    std::iota(std::begin(indices), std::end(indices), 0);

    std::vector<double> data(num_multiplier_dofs_, 1.0);

    return SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                        num_edge_dofs_, num_multiplier_dofs_);
}

void HybridSolver::Solve(const BlockVector& Rhs, BlockVector& Sol) const
{
    RHSTransform(Rhs, Hrhs_);

    if (!use_w_ && myid_ == 0)
    {
        Hrhs_[0] = 0.0;
    }

    // assemble true right hand side
    multiplier_d_td_.MultAT(Hrhs_, trueHrhs_);

    // solve the parallel global hybridized system
    Timer timer(Timer::Start::True);

    trueMu_ = 0.0;
    cg_.Mult(trueHrhs_, trueMu_);

    timer.Click();
    timing_ += timer.TotalTime();

    if (myid_ == 0 && print_level_ > 0)
    {
        std::cout << "  Timing: PCG done in " << timing_ << "s. \n";
    }

    num_iterations_ += cg_.GetNumIterations();

    if (myid_ == 0 && print_level_ > 0)
    {
        /*
        if (cg_.GetConverged())
            std::cout << "  CG converged in "
                      << num_iterations_
                      << " with a final residual norm "
                      << cg_->GetFinalNorm() << "\n";
        else
            std::cout << "  CG did not converge in "
                      << num_iterations_
                      << ". Final residual norm is "
                      << cg_->GetFinalNorm() << "\n";
        */
    }

    // distribute true dofs to dofs and recover solution of the original system
    multiplier_d_td_.Mult(trueMu_, Mu_);
    RecoverOriginalSolution(Mu_, Sol);
}

void HybridSolver::RHSTransform(const BlockVector& OriginalRHS,
                                VectorView HybridRHS) const
{
    HybridRHS = 0.;

    Vector f_loc;
    Vector CMinvDTAinv_f_loc;

    for (int iAgg = 0; iAgg < num_aggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        std::vector<int> local_vertexdof = agg_vertexdof_.GetIndices(iAgg);
        std::vector<int> local_multiplier = agg_multiplier_.GetIndices(iAgg);

        int nlocal_vertexdof = local_vertexdof.size();
        int nlocal_multiplier = local_multiplier.size();

        // Compute local contribution to the RHS of the hybrid system
        OriginalRHS.GetBlock(1).GetSubVector(local_vertexdof, f_loc);
        f_loc *= -1.0;

        CMinvDTAinv_f_loc.SetSize(nlocal_multiplier);
        AinvDMinvCT_[iAgg].MultAT(f_loc, CMinvDTAinv_f_loc);

        for (int i = 0; i < nlocal_multiplier; ++i)
        {
            HybridRHS[local_multiplier[i]] -= CMinvDTAinv_f_loc[i];
        }

        // Save the element rhs [M B^T;B 0]^-1[f;g] for solution recovery
        Ainv_f_[iAgg].SetSize(nlocal_vertexdof);
        Ainv_[iAgg].Mult(f_loc, Ainv_f_[iAgg]);
        Ainv_f_[iAgg] /= agg_weights_[iAgg];
    }
}

void HybridSolver::RecoverOriginalSolution(const VectorView& HybridSol,
                                           BlockVector& RecoveredSol) const
{
    // Recover the solution of the original system from multiplier mu, i.e.,
    // [u;p] = [f;g] - [M B^T;B 0]^-1[C 0]^T * mu
    // This procedure is done locally in each element

    RecoveredSol = 0.;

    Vector mu_loc;
    Vector sigma_loc;
    Vector tmp;

    for (int iAgg = 0; iAgg < num_aggs_; ++iAgg)
    {
        // Extracting the size and global numbering of local dof
        std::vector<int> local_vertexdof = agg_vertexdof_.GetIndices(iAgg);
        std::vector<int> local_edgedof = agg_edgedof_.GetIndices(iAgg);
        std::vector<int> local_multiplier = agg_multiplier_.GetIndices(iAgg);

        int nlocal_vertexdof = local_vertexdof.size();
        int nlocal_edgedof = local_edgedof.size();
        int nlocal_multiplier = local_multiplier.size();

        // Initialize a vector which will store the local contribution of Hdiv
        // and L2 space
        Vector& u_loc(Ainv_f_[iAgg]);

        // This check is just for the case when there is only one element for
        // the global problem, then there will be no Lagrange multipliers
        if (nlocal_multiplier > 0)
        {
            // Extract the local portion of the Lagrange multiplier solution
            HybridSol.GetSubVector(local_multiplier, mu_loc);

            // Compute u = (DMinvDT)^-1(f-DMinvC^T mu)
            //AinvDMinvCT_[iAgg].AddMult_a(-1., mu_loc, u_loc);
            tmp.SetSize(u_loc.size());
            AinvDMinvCT_[iAgg].Mult(mu_loc, tmp);
            u_loc -= tmp;

            // Compute -sigma = Minv(DT u + DT mu)
            sigma_loc.SetSize(nlocal_edgedof);
            MinvDT_[iAgg].Mult(u_loc, sigma_loc);
            //MinvCT_[iAgg].AddMult(mu_loc, sigma_loc);
            tmp.SetSize(sigma_loc.size());
            MinvCT_[iAgg].Mult(mu_loc, tmp);
            sigma_loc += tmp;
            sigma_loc *= agg_weights_[iAgg];
        }

        // Save local solution to the global solution vector
        for (int i = 0; i < nlocal_edgedof; ++i)
        {
            RecoveredSol[local_edgedof[i]] = -sigma_loc[i];
        }

        for (int i = 0; i < nlocal_vertexdof; ++i)
        {
            RecoveredSol.GetBlock(1)[local_vertexdof[i]] = u_loc[i];
        }
    }
}

void HybridSolver::UpdateAggScaling(const std::vector<double>& agg_weight)
{
    assert(!use_w_);
    assert(static_cast<int>(agg_weight.size()) == num_aggs_);

    std::copy(std::begin(agg_weight), std::end(agg_weight), std::begin(agg_weights_));

    CooMatrix hybrid_system(num_multiplier_dofs_);

    for (int i = 0; i < num_aggs_; ++i)
    {
        std::vector<int> local_multipliers = agg_multiplier_.GetIndices(i);

        hybrid_system.Add(local_multipliers, agg_weight[i], hybrid_elem_[i]);
    }

    InitSolver(hybrid_system.ToSparse());
}

void HybridSolver::SetPrintLevel(int print_level)
{
    MGLSolver::SetPrintLevel(print_level);

    if (myid_ == 0)
    {
        cg_.SetVerbose(print_level_);
    }
}

void HybridSolver::SetMaxIter(int max_num_iter)
{
    MGLSolver::SetMaxIter(max_num_iter);

    cg_.SetMaxIter(max_num_iter_);
}

void HybridSolver::SetRelTol(double rtol)
{
    MGLSolver::SetRelTol(rtol);

    cg_.SetRelTol(rtol_);
}

void HybridSolver::SetAbsTol(double atol)
{
    MGLSolver::SetAbsTol(atol);

    cg_.SetAbsTol(atol_);
}

} // namespace smoothg
