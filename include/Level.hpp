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

namespace gauss
{

/**
   @brief Collection of data per level
*/
struct Level
{
    Level() = default;
    Level(const Graph& graph, std::vector<int> elim_dofs_in = {});
    Level(MixedMatrix mm, GraphSpace gs, Vector const_vect,
          bool assemble_M = true, std::vector<int> elim_dofs_in = {});

    MixedMatrix mixed_matrix;
    GraphSpace graph_space;
    Vector constant_rep;

    std::unique_ptr<MGLSolver> solver;
    mutable BlockVector rhs;
    mutable BlockVector sol;

    std::vector<int> edge_elim_dofs;
};

inline
Level::Level(const Graph& graph, std::vector<int> elim_dofs_in)
    : Level(MixedMatrix(graph), FineGraphSpace(graph),
            Vector(graph.vertex_edge_local_.Rows(), 1.0 / std::sqrt(graph.global_vertices_)),
            true, std::move(elim_dofs_in))
{

}

inline
Level::Level(MixedMatrix mm, GraphSpace gs, Vector const_vect,
             bool assemble_M, std::vector<int> elim_dofs_in)
    : mixed_matrix(std::move(mm)), graph_space(std::move(gs)),
      constant_rep(std::move(const_vect)),
      rhs(mixed_matrix.Offsets()), sol(mixed_matrix.Offsets()),
      edge_elim_dofs(std::move(elim_dofs_in))
{
    if (assemble_M)
    {
        mixed_matrix.AssembleM();
    }
}


} // namespace gauss

#endif /* LEVEL_HPP */
