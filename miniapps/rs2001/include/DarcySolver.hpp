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

#ifndef DARCYSOLVER_HPP_
#define DARCYSOLVER_HPP_

#include <memory>

#include "spe10.hpp"

namespace rs2000
{
/// \class DarcySolver
/// \brief Constructs the DeRham sequence to solve Darcy's Equation
///
/// Assembles the finite element matrices for the Darcy operator
///
///                           D = [ M(k)  B^T ]
///                               [ B     0   ]
///    where:
///
///    M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
///    B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
/// The solver/preconditioner strategy is specified via the ParameterList.
/// M(k) and the preconditioner is recomputed for each input k.

class DarcySolver
{
public:
    DarcySolver(smoothg::GraphUpscale& upscale);

    void SetRHS(const smoothg::BlockVector& fine_rhs);
    void SetObsFunc(const smoothg::BlockVector& fine_obs, double area);

    /// Solve and update quantity of interest Q, cost C
    void SolveFwd(
        int ilevel,
        mfem::Vector& k_over_k_ref,
        double& Q,
        double& C);

    int GetNumberOfDofs(int ilevel) const { return upscale_.GetMatrix(ilevel).Rows(); }

    int GetGlobalNumberOfDofs(int ilevel) const { return upscale_.GetMatrix(ilevel).GlobalRows(); }

    int GetNNZ(int ilevel) const {return upscale_.Solver(ilevel).GetNNZ(); }

    double GetSolveTime(int ilevel) const {return upscale_.SolveTime(ilevel); }

    const smoothg::BlockVector& Solution(int ilevel) const { return sol_[ilevel]; }

    void InterpolateCoeff(int level, smoothg::Vector& coeff);

private:
    smoothg::GraphUpscale& upscale_;
    std::vector<smoothg::BlockVector> rhs_;
    std::vector<smoothg::BlockVector> sol_;
    std::vector<smoothg::Vector> constant_rep_;
    std::vector<smoothg::BlockVector> obs_;


    std::vector<std::vector<double>> coeff_;
    std::vector<std::vector<int>> coeff_marker_;

    double size_bndr_;
};

} /* namespace rs2000 */
#endif /* DARCYSOLVER_HPP_ */

