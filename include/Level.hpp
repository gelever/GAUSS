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

    @brief Contains Level class
*/

#ifndef LEVEL_HPP
#define LEVEL_HPP

#include "Utilities.hpp"
#include "MixedMatrix.hpp"
#include "MGLSolver.hpp"
#include "GraphSpace.hpp"

#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"
#include "SPDSolver.hpp"

namespace smoothg
{

/**
   @brief Collection of data per level
*/
struct Level
{
    Level() = default;
    Level(MixedMatrix mm, GraphSpace gs, bool assemble_M = true);
    Level(MixedMatrix mm, GraphSpace gs, Vector const_vect, bool assemble_M = true);

    MixedMatrix mixed_matrix;
    GraphSpace graph_space;
    Vector constant_rep;

    std::unique_ptr<MGLSolver> solver;
    mutable BlockVector rhs;
    mutable BlockVector sol;

    std::vector<int> elim_dofs;
};

inline
Level::Level(MixedMatrix mm, GraphSpace gs, bool assemble_M)
: mixed_matrix(std::move(mm)), graph_space(std::move(gs)),
  constant_rep(mixed_matrix.LocalD().Rows(), 1.0 / std::sqrt(mixed_matrix.GlobalD().GlobalRows())),
  rhs(mixed_matrix.Offsets()), sol(mixed_matrix.Offsets())
{
    if (assemble_M)
    {
        mixed_matrix.AssembleM();
    }
}

inline
Level::Level(MixedMatrix mm, GraphSpace gs, Vector const_vect, bool assemble_M)
: mixed_matrix(std::move(mm)), graph_space(std::move(gs)),
  constant_rep(std::move(const_vect)),
  rhs(mixed_matrix.Offsets()), sol(mixed_matrix.Offsets())
{
    if (assemble_M)
    {
        mixed_matrix.AssembleM();
    }
}


} // namespace smoothg

#endif /* LEVEL_HPP */
