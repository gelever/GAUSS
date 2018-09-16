// Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-XXXXXX. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the ParELAGMC library. For more information and
// source code availability see http://github.com/LLNL/parelagmc.
//
// ParELAGMC is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License (as published by the
// Free Software Foundation) version 2.1 dated February 1999.

#include "DarcySolver.hpp"

namespace rs2000
{

DarcySolver::DarcySolver(smoothg::GraphUpscale& upscale)
    : upscale_(upscale),
      rhs_(upscale_.GetMLBlockVector()),
      sol_(upscale_.GetMLBlockVector()),
      constant_rep_(upscale_.GetMLVectors()),
      obs_(upscale_.GetMLBlockVector()),
      coeff_(upscale_.NumLevels()),
      coeff_marker_(upscale_.NumLevels()),
      size_bndr_(1.0)
{
    int num_levels = upscale_.NumLevels();

    constant_rep_[0] = 1.0;

    for (int i = 0; i < num_levels - 1; ++i)
    {
        upscale_.Coarsener(i).Restrict(constant_rep_[i], constant_rep_[i + 1]);
    }

    for (int i = 0; i < num_levels; ++i)
    {
        obs_[i] = 0.0;

        int num_elem = upscale_.GetMatrix(i).GetElemDof().Rows();
        int num_dofs = constant_rep_[i].size();

        coeff_[i].resize(num_elem, -1);

        for (int j = 0; j < num_dofs; ++j)
        {
            if (std::fabs(constant_rep_[i][j]) > (1.0 - 1e-8))
            {
                coeff_marker_[i].push_back(j);
            }
        }

        assert(static_cast<int>(coeff_marker_[i].size()) == num_elem);
    }
}

void DarcySolver::SetObsFunc(const smoothg::BlockVector& fine_obs, double area)
{
    size_bndr_ = area;

    int num_levels = upscale_.NumLevels();

    obs_[0] = fine_obs;

    for (int i = 0; i < num_levels - 1; ++i)
    {
        upscale_.Coarsener(i).Restrict(obs_[i], obs_[i + 1]);
    }
}

void DarcySolver::SetRHS(const smoothg::BlockVector& fine_rhs)
{
    int num_levels = upscale_.NumLevels();

    rhs_[0] = fine_rhs;

    for (int i = 0; i < num_levels - 1; ++i)
    {
        upscale_.Coarsener(i).Restrict(rhs_[i], rhs_[i + 1]);
    }
}

void DarcySolver::SolveFwd(
    int ilevel,
    mfem::Vector& k_over_k_ref,
    double& Q,
    double& C)
{
    int num_elem = coeff_[ilevel].size();
    assert(k_over_k_ref.Size() == num_elem);

    for (int i = 0; i < num_elem; ++i)
    {
        coeff_[ilevel][i] = k_over_k_ref[i];
    }

    upscale_.RescaleSolver(ilevel, coeff_[ilevel]);
    upscale_.SolveLevel(ilevel, rhs_[ilevel], sol_[ilevel]);

    upscale_.Orthogonalize(ilevel, sol_[ilevel]);

    MPI_Comm comm = upscale_.GetMatrix(ilevel).GlobalD().GetComm();

    //C = upscale_.GetMatrix(ilevel).GlobalRows();
    C = upscale_.Solver(ilevel).GetTiming();

    Q = linalgcpp::ParMult(comm, obs_[ilevel], sol_[ilevel]) / size_bndr_;
}

void DarcySolver::InterpolateCoeff(int level, smoothg::Vector& coeff)
{
    int coarse_size = constant_rep_[level].size();
    int fine_size = constant_rep_[0].size();

    smoothg::Vector vect(coarse_size, 0.0);

    int coeff_size = coeff_[level].size();

    for (int i = 0; i < coeff_size; ++i)
    {
        int index = coeff_marker_[level][i];

        vect[index] = coeff_[level][i];
    }

    vect *= constant_rep_[level];

    coeff.SetSize(fine_size);
    upscale_.Interpolate(vect, coeff);
}

}
