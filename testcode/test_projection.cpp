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
   @file test_projection.cpp
   @brief tests projection properties

   The following should hold, where pi is the projector
   * D (pi_sigma) = (pi_u) D
   * pi pi = pi
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // Graph Params
    int gen_vertices = 400;
    int mean_degree = 10;
    double beta = 0.15;
    int seed = -1;
    int coarsen_factor = 40;

    // Upscale Params
    int max_evects = 2;
    double spect_tol = 1;

    // Test Params
    bool failed = false;
    double test_tol = 1e-10;

    /// [Load Input]
    SparseMatrix vertex_edge_global = GenerateGraph(comm, gen_vertices, mean_degree, beta, seed);
    std::vector<int> global_partitioning = PartitionAAT(vertex_edge_global, coarsen_factor);
    /// [Load Input]

    // Set up GraphUpscale
    /// [Upscale]
    Graph graph(comm, vertex_edge_global, global_partitioning);
    GraphUpscale upscale(std::move(graph), spect_tol, max_evects);

    upscale.PrintInfo();
    upscale.ShowSetupTime();
    /// [Upscale]

    /// [Test Projection]
    BlockVector test_vect = upscale.GetBlockVector(0);
    test_vect.GetBlock(0).Randomize(-1.0, 1.0);
    test_vect.GetBlock(0).Normalize();

    const auto& D = upscale.GetMatrix(0).LocalD();

    D.Mult(test_vect.GetBlock(0), test_vect.GetBlock(1));

    auto proj_coarse = upscale.Project(test_vect);
    auto proj_fine = upscale.Interpolate(proj_coarse);

    {
        auto D_proj = D.Mult(proj_fine.GetBlock(0));
        auto D_error = CompareError(comm, D_proj, proj_fine.GetBlock(1));

        ParPrint(myid, std::cout << "D Projection Error:  " << D_error << "\n");

        failed |= std::fabs(D_error) > test_tol;
    }

    {
        upscale.Project(proj_fine, proj_coarse);
        auto re_proj_fine = upscale.Interpolate(proj_coarse);

        auto reproject_error = CompareError(comm, re_proj_fine, proj_fine);

        ParPrint(myid, std::cout << "Re Projection Error: " << reproject_error << "\n");

        failed |= std::fabs(reproject_error) > test_tol;
    }

    /// [Test Projection]

    return failed;
}
