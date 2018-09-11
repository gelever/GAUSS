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
   @file mltopo.cpp
   @brief This is an example showing how to generate a hierarchy of graphs by
   recursive coarsening.

   A simple way to run the example:

   ./mltopo
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "spe10.hpp"

using namespace rs2000;
using smoothg::Timer;

void ShowAggregates(std::vector<std::vector<int>> parts,
                    mfem::ParMesh& pmesh);

int main(int argc, char* argv[])
{
    // 1. Initialize MPI
    smoothg::MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    mfem::OptionsParser args(argc, argv);
    const char* permFile = "spe_perm.dat";
    args.AddOption(&permFile, "-p", "--perm",
                   "SPE10 permeability file data.");
    int nDimensions = 2;
    args.AddOption(&nDimensions, "-d", "--dim",
                   "Dimension of the physical space.");
    bool metis_agglomeration = false;
    args.AddOption(&metis_agglomeration, "-ma", "--metis-agglomeration",
                   "-nm", "--no-metis-agglomeration",
                   "Use Metis as the partitioner (instead of geometric).");
    double proc_part_ubal = 2.0;
    args.AddOption(&proc_part_ubal, "-pub", "--part-unbalance",
                   "Processor partition unbalance factor.");
    int spe10_scale = 5;
    args.AddOption(&spe10_scale, "-sc", "--spe10-scale",
                   "Scale of problem, 1=small, 5=full SPE10.");
    bool visualization = false;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable visualization.");
    int num_refine = 0;
    args.AddOption(&num_refine, "-nr", "--num-refine",
                   "Number of mesh refinements.");
    double coarse_factor = 1.0;
    args.AddOption(&coarse_factor, "-cf", "--coarse-factor",
                   "Scale coarsening factor.");
    int num_levels = 4;
    args.AddOption(&num_levels, "-nl", "--num-levels",
                   "Number of levels to coarsen.");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(std::cout);
    }

    mfem::Array<double> coarseningFactor(nDimensions);
    coarseningFactor[0] = 10;
    coarseningFactor[1] = 10;
    if (nDimensions == 3)
        coarseningFactor[2] = 5;

    // Setting up finite volume discretization problem
    int slice = 0;
    bool metis_proc_part = false;
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_proc_part, proc_part_ubal, coarseningFactor);

    mfem::ParMesh* pmesh = spe10problem.GetParMesh();

    // Visualize aggregates in all levels
    if (visualization)
    {
        int num_parts = 3;

        std::vector<std::vector<int>> parts;
        for (int i = 0; i < num_parts; ++i)
        {
            parts.push_back(linalgcpp::ReadText<int>("/home/gelever1/Code/mylibs/mysmoothg/build/examples/topo_part_0000" + std::to_string(i) + ".txt"));

        }

        ShowAggregates(parts, *pmesh);
    }

    return EXIT_SUCCESS;
}

void ShowAggregates(std::vector<std::vector<int>> parts,
                    mfem::ParMesh& pmesh)
{
    mfem::L2_FECollection attr_fec(0, pmesh.SpaceDimension());
    mfem::ParFiniteElementSpace attr_fespace(&pmesh, &attr_fec);
    mfem::ParGridFunction attr(&attr_fespace);

    mfem::socketstream sol_sock;

    for (int i = 0; i < parts.size(); ++i)
    {
        for (int j = 0; j < parts[i].size(); j++)
        {
            attr[j] = parts[i][j];
        }

        char vishost[] = "localhost";
        int  visport   = 19916;
        sol_sock.open(vishost, visport);

        if (sol_sock.is_open())
        {
            sol_sock.precision(8);
            sol_sock << "parallel " << pmesh.GetNRanks() << " " << pmesh.GetMyRank() << "\n";
            if (pmesh.SpaceDimension() == 2)
            {
                sol_sock << "fem2d_gf_data_keys\n";
            }
            else
            {
                sol_sock << "fem3d_gf_data_keys\n";
            }

            pmesh.PrintWithPartitioning(parts[i].data(), sol_sock, 0);
            attr.Save(sol_sock);

            sol_sock << "window_size 500 800\n";
            sol_sock << "window_title 'Level " << i + 1 << " aggregation'\n";

            if (pmesh.SpaceDimension() == 2)
            {
                sol_sock << "view 0 0\n"; // view from top
                sol_sock << "keys jl\n";  // turn off perspective and light
                sol_sock << "keys ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]\n";  // increase size
                sol_sock << "keys b\n";  // draw interface
            }
            else
            {
                sol_sock << "keys ]]]]]]]]]]]]]\n";  // increase size
            }

            MPI_Barrier(pmesh.GetComm());
        }
    }
}
