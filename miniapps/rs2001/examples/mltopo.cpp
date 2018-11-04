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

using namespace rs2001;
using gauss::Timer;

void ShowAggregates(const std::vector<gauss::GraphTopology>& graph_topos,
                    mfem::ParMesh& pmesh);

int main(int argc, char* argv[])
{
    // 1. Initialize MPI
    gauss::MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm;
    int myid = mpi_info.myid;
    int num_procs = mpi_info.num_procs;

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

    for (int i = 0; i < num_refine; ++i)
    {
        pmesh->UniformRefinement();
    }

    if (myid == 0)
    {
        std::cout << pmesh->GetNEdges() << " fine edges, " <<
                  pmesh->GetNFaces() << " fine faces, " <<
                  pmesh->GetNE() << " fine elements\n";
    }

    // Construct "finite volume mass" matrix using mfem instead of parelag
    mfem::RT_FECollection sigmafec(0, nDimensions);
    mfem::ParFiniteElementSpace sigmafespace(pmesh, &sigmafec);

    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::ParFiniteElementSpace ufespace(pmesh, &ufec);

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                              : pmesh->ElementToFaceTable();
    gauss::SparseMatrix vertex_edge = TableToSparse(vertex_edge_table);

    // Construct agglomerated topology based on METIS or Cartesion aggloemration
    for (auto&& i : coarseningFactor)
    {
        i *= coarse_factor;
    }

    std::vector<int> partition;
    if (metis_agglomeration)
    {
        partition = MetisPart(sigmafespace, ufespace, coarseningFactor);
    }
    else
    {
        auto num_procs_xyz = spe10problem.GetNumProcsXYZ();
        partition = CartPart(num_procs_xyz, *pmesh, coarseningFactor);
    }

    gauss::ParMatrix edge_d_td = ParMatrixToParMatrix(*sigmafespace.Dof_TrueDof_Matrix());

    // Create Fine Level Topology
    std::vector<gauss::GraphTopology> topos;
    topos.emplace_back(gauss::Graph(vertex_edge, edge_d_td, partition));

    // Build multilevel graph topology
    double coarsen_factor = (nDimensions  == 2) ? 8 : 32;
    for (int i = 1; i < num_levels; ++i)
    {
        topos.emplace_back(topos.back(), coarsen_factor);
    }

    // Visualize aggregates in all levels
    if (visualization)
    {
        ShowAggregates(topos, *pmesh);
    }

    return EXIT_SUCCESS;
}

void ShowAggregates(const std::vector<gauss::GraphTopology>& topos,
                    mfem::ParMesh& pmesh)
{
    mfem::L2_FECollection attr_fec(0, pmesh.SpaceDimension());
    mfem::ParFiniteElementSpace attr_fespace(&pmesh, &attr_fec);
    mfem::ParGridFunction attr(&attr_fespace);

    mfem::socketstream sol_sock;

    for (int i = 0; i < topos.size(); ++i)
    {
        gauss::SparseMatrix agg_vertex = topos[0].agg_vertex_local_;

        for (int j = 1; j < i + 1; ++j)
        {
            agg_vertex = topos[j].agg_vertex_local_.Mult(agg_vertex);
        }

        gauss::SparseMatrix vertex_agg = agg_vertex.Transpose();

        const gauss::SparseMatrix& agg_face = topos[i].agg_face_local_;
        const gauss::SparseMatrix& face_agg = topos[i].face_agg_local_;
        gauss::SparseMatrix agg_agg = agg_face.Mult(face_agg);

        std::vector<int> color = gauss::GetElementColoring(agg_agg);

        int num_colors = *std::max_element(std::begin(color), std::end(color)) + 1;
        num_colors = std::max(num_colors, pmesh.GetNRanks());

        auto& partitioning = vertex_agg.GetIndices();

        for (int j = 0; j < vertex_agg.Rows(); j++)
        {
            attr[j] = (color[partitioning[j]] + pmesh.GetMyRank()) % num_colors;
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

            pmesh.PrintWithPartitioning(partitioning.data(), sol_sock, 0);
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
