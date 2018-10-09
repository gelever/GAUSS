/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-759464. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of GAUSS. For more information and source code
 * availability, see https://www.github.com/gelever/GAUSS.
 *
 * GAUSS is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/**
   @file Sampler.cpp
   @brief Contains sampler implementations
*/

#include "Sampler.hpp"

namespace rs2001
{

double ScalingCoeff(double corlen, double myDim)
{
    const double nu = 2.0 - (myDim / 2.0);
    const double gnu = std::tgamma(nu);
    const double gnudim = std::tgamma(nu + myDim);
    const double c = std::pow(16.0 * std::atan(1.0), 0.5 * myDim);
    const double k = std::pow(1.0 / corlen, 2.0 * nu);

    return std::sqrt(c * gnudim * k / gnu);
}

PDESampler::PDESampler(const gauss::Graph& graph, const gauss::UpscaleParams& params,
                       int dimension, double corlen, double cell_volume, bool lognormal)
    : upscale_(graph, params),
      normal_dist_(0.0, 1.0, graph.edge_true_edge_.GetMyId()),
      cell_volume_(cell_volume),
      alpha_(1.0 / (corlen * corlen)),
      matern_coeff_(ScalingCoeff(corlen, dimension)),
      log_normal_(lognormal),
      constant_map_(upscale_.NumLevels()),
      rhs_(upscale_.GetMLVectors()),
      sol_(upscale_.GetMLVectors()),
      constant_rep_(upscale_.GetMLVectors()),
      Ws_(upscale_.NumLevels()),
      coefficient_(upscale_.NumLevels()),
      upscaled_coeff_(upscale_.NumLevels(), upscale_.GetVector(0)),
      solve_iters_(upscale_.NumLevels(), 0),
      solve_time_(upscale_.NumLevels(), 0.0),
      global_sample_size_(upscale_.NumLevels())
{
    upscale_.PrintInfo();
    upscale_.ShowSetupTime();

    MPI_Comm comm = upscale_.GetMatrix(0).GlobalD().GetComm();

    int num_procs;
    int myid;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &num_procs);

    // Compute the (not normalized) constant vector
    constant_rep_[0] = 1.0;

    for (int i = 0; i < upscale_.NumLevels() - 1; ++i)
    {
        upscale_.Coarsener(i).Restrict(constant_rep_[i], constant_rep_[i + 1]);
    }

    gauss::SparseMatrix W = gauss::SparseIdentity(constant_rep_[0].size());
    W = cell_volume_;


    for (int i = 0; i < upscale_.NumLevels(); ++i)
    {
        Ws_[i] = W.GetDiag();

        int size = Ws_[i].size();

        for (int j = 0; j < size; ++j)
        {
            assert(Ws_[i][j] > 0);
            Ws_[i][j] = std::sqrt(Ws_[i][j]);
        }

        if (i < upscale_.NumLevels() - 1)
        {
            const auto& P = upscale_.Coarsener(i).Pvertex();
            gauss::SparseMatrix PT = P.Transpose();
            W = PT.Mult(W).Mult(P);
        }
    }

    for (int i = 0; i < upscale_.NumLevels(); ++i)
    {
        coefficient_[i].resize(upscale_.GetMatrix(i).GetElemDof().Rows());
    }

    for (int i = 0; i < upscale_.NumLevels(); ++i)
    {
        size_to_level_[upscale_.GetMatrix(i).LocalD().Rows()] = i;
    }

    for (int i = 0; i < upscale_.NumLevels(); ++i)
    {
        int size = constant_rep_[i].size();

        for (int j = 0; j < size; ++j)
        {
            if (std::fabs(constant_rep_[i][j]) > (1.0 - 1e-8))
            {
                constant_map_[i].push_back(j);
            }
        }

        assert(static_cast<int>(constant_map_[i].size()) == upscale_.GetMatrix(i).GetElemDof().Rows());
    }

    for (int i = 0; i < upscale_.NumLevels(); ++i)
    {
        int local_size = constant_rep_[i].size();
        int global_size;

        MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);
        global_sample_size_[i] = global_size;
    }
}

void PDESampler::Sample(bool )
{
    throw std::runtime_error("Old Interface no longer implemented!");
}

void PDESampler::Sample(const int level, mfem::Vector& xi)
{
    int size = Ws_[level].size();
    xi.SetSize(size);

    normal_dist_(xi);

    for (int i = 0; i < size; ++i)
    {
        if (std::fabs(constant_rep_[level][i]) < 1.0 - 1e-8)
        {
            xi[i] = 0.0;
        }
        else
        {
            xi[i] *= matern_coeff_ * Ws_[level][i];

        }
    }
}

void PDESampler::Eval(const int level, const mfem::Vector& xi, mfem::Vector& s)
{
    int xi_level = size_to_level_.at(xi.Size());
    int xi_size = xi.Size();

    assert(xi_level <= level);
    assert(xi_size == rhs_[xi_level].size());

    rhs_[xi_level] = 0.0;

    for (int i = 0; i < xi_size; ++i)
    {
        rhs_[xi_level][i] = xi[i];
    }

    upscale_.Restrict(rhs_[xi_level], rhs_[level]);
    upscale_.SolveLevel(level, rhs_[level], sol_[level]);

    const int size_s = upscale_.GetMatrix(level).GetElemDof().Rows();

    s.SetSize(size_s);

    for (int i = 0; i < size_s; ++i)
    {
        int index = constant_map_[level][i];

        s[i] = sol_[level][index] / constant_rep_[level][index];
    }


    if (log_normal_)
    {
        for (int i = 0; i < size_s; ++i)
        {
            s[i] = std::exp(s[i]);
        }
    }

}

void PDESampler::Eval(const int level, const mfem::Vector& xi, mfem::Vector& s,
                      mfem::Vector& u, bool use_init)
{
    assert(!use_init);
    //TODO(gelever1): fix this proper to us initial guess
    Eval(level, xi, s);
    u = s;
}

double PDESampler::ComputeL2Error(int level, const mfem::Vector& coeff, double exact) const
{
    assert(fespace_);
    mfem::Vector coeff_fine = rs2001::VectorToVector(Interpolate(level, coeff));

    mfem::GridFunction x;
    x.MakeRef(fespace_, coeff_fine, 0);

    mfem::ConstantCoefficient exact_soln(exact);
    const double err = x.ComputeL2Error(exact_soln);

    return err * err;
}

gauss::Vector PDESampler::Interpolate(int level, const mfem::Vector& coeff) const
{
    ///*
    gauss::Vector vect(rhs_[level].size(), 0.0);

    int size = coeff.Size();

    for (int i = 0; i < size; ++i)
    {
        int index = constant_map_[level][i];

        vect[index] = coeff[i];
    }
    //*/

    //gauss::Vector vect = VectorToVector(coeff);
    vect *= constant_rep_[level];

    auto vect_fine = upscale_.Interpolate(vect, 0);
    //upscale_.Orthogonalize(0, vect_fine);

    return vect_fine;
}

mfem::Vector PDESampler::Restrict(int level, const mfem::Vector& coeff) const
{
    gauss::Vector fine_vect = rs2001::VectorToVector(coeff);
    gauss::Vector coarse_vect = upscale_.Restrict(fine_vect, level);

    int size = constant_map_[level].size();
    mfem::Vector vect(size);
    vect = 0.0;

    for (int i = 0; i < size; ++i)
    {
        int index = constant_map_[level][i];

        vect[i] = coarse_vect[index] / constant_rep_[level][index];
    }

    return vect;
}


} // namespace rs2001
