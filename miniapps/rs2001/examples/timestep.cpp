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
   @file timestep.cpp
   @brief Visualized pressure over time of a simple reservior model.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "spe10.hpp"

using namespace rs2000;
using smoothg::Timer;

smoothg::Vector InitialCondition(mfem::ParFiniteElementSpace& ufespace, double initial_val);

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
    int slice = 0;
    args.AddOption(&slice, "-s", "--slice",
                   "Slice of SPE10 data to take for 2D run.");
    int max_evects = 4;
    args.AddOption(&max_evects, "-m", "--max-evects",
                   "Maximum eigenvectors per aggregate.");
    double spect_tol = 1.e-3;
    args.AddOption(&spect_tol, "-t", "--spect-tol",
                   "Spectral tolerance for eigenvalue problems.");
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
    bool hybridization = false;
    args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                   "--no-hybridization", "Enable hybridization.");
    int num_refine = 0;
    args.AddOption(&num_refine, "-nr", "--num-refine",
                   "Number of time to refine mesh.");
    double delta_t = 10.0;
    args.AddOption(&delta_t, "-dt", "--delta-t",
                   "Time step.");
    double total_time = 10000.0;
    args.AddOption(&total_time, "-time", "--total-time",
                   "Total time to step.");
    double initial_val = 1.0;
    args.AddOption(&initial_val, "-iv", "--initial-value",
                   "Initial pressure difference.");
    int vis_step = 0;
    args.AddOption(&vis_step, "-vs", "--vis-step",
                   "Step size for visualization.");
    bool dual_target = false;
    args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = false;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = false;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    const char* caption = "";
    args.AddOption(&caption, "-cap", "--caption",
                   "Caption for visualization");
    double coarse_factor = 1.0;
    args.AddOption(&coarse_factor, "-cf", "--coarse-factor",
                   "Scale coarsening factor.");
    int num_levels = 2;
    args.AddOption(&num_levels, "-nl", "--num-levels",
                   "Number of levels.");
    double ml_factor = 4.0;
    args.AddOption(&ml_factor, "-mf", "--ml-factor",
                   "Multilevel aggregate coarsening factor.");
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

    int nbdr;
    if (nDimensions == 3)
        nbdr = 6;
    else
        nbdr = 4;
    mfem::Array<int> ess_zeros(nbdr);
    mfem::Array<int> nat_one(nbdr);
    mfem::Array<int> nat_zeros(nbdr);
    ess_zeros = 1;
    nat_one = 0;
    nat_zeros = 0;

    mfem::Array<int> ess_attr;
    mfem::Vector mfem_weight;
    mfem::Vector rhs_u_fine;

    // Setting up finite volume discretization problem
    SPE10Problem spe10problem(permFile, nDimensions, spe10_scale, slice,
                              metis_agglomeration, proc_part_ubal, coarseningFactor);

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

    ess_attr.SetSize(nbdr);
    for (int i(0); i < nbdr; ++i)
        ess_attr[i] = ess_zeros[i];

    // Construct "finite volume mass" matrix using mfem instead of parelag
    mfem::RT_FECollection sigmafec(0, nDimensions);
    mfem::ParFiniteElementSpace sigmafespace(pmesh, &sigmafec);

    mfem::ParBilinearForm a(&sigmafespace);
    a.AddDomainIntegrator(
        new FiniteVolumeMassIntegrator(*spe10problem.GetKInv()) );
    a.Assemble();
    a.Finalize();
    a.SpMat().GetDiag(mfem_weight);

    mfem::L2_FECollection ufec(0, nDimensions);
    mfem::ParFiniteElementSpace ufespace(pmesh, &ufec);

    mfem::LinearForm q(&ufespace);
    q.AddDomainIntegrator(
        new mfem::DomainLFIntegrator(*spe10problem.GetForceCoeff()) );
    q.Assemble();
    rhs_u_fine = q;

    // Construct vertex_edge table in mfem::SparseMatrix format
    auto& vertex_edge_table = nDimensions == 2 ? pmesh->ElementToEdgeTable()
                              : pmesh->ElementToFaceTable();
    smoothg::SparseMatrix vertex_edge = TableToSparse(vertex_edge_table);

    // Construct agglomerated topology based on METIS or Cartesion aggloemration
    for (auto&& i : coarseningFactor)
    {
        i *= coarse_factor;
    }

    std::vector<int> partitioning;
    if (metis_agglomeration)
    {
        partitioning = MetisPart(sigmafespace, ufespace, coarseningFactor);
    }
    else
    {
        auto num_procs_xyz = spe10problem.GetNumProcsXYZ();
        partitioning = CartPart(num_procs_xyz, *pmesh, coarseningFactor);
    }

    smoothg::ParMatrix edge_d_td = ParMatrixToParMatrix(*sigmafespace.Dof_TrueDof_Matrix());
    smoothg::SparseMatrix edge_boundary_att = GenerateBoundaryAttributeTable(*pmesh);
    smoothg::SparseMatrix boundary_att_edge = edge_boundary_att.Transpose();

    std::vector<int> elim_edges;
    int num_bdr = boundary_att_edge.Rows();

    for (int i = 0; i < num_bdr; ++i)
    {
        std::vector<int> edges = boundary_att_edge.GetIndices(i);
        elim_edges.insert(std::end(elim_edges), std::begin(edges),
                          std::end(edges));
    }

    for (auto&& edge : elim_edges)
    {
        mfem_weight[edge] *= 2.0;
    }

    std::vector<double> weight(mfem_weight.Size());
    for (int i = 0; i < mfem_weight.Size(); ++i)
    {
        weight[i] = 1.0 / mfem_weight[i];
    }

    smoothg::SparseMatrix W_block = smoothg::SparseIdentity(vertex_edge.Rows());

    const double cell_volume = spe10problem.CellVolume(nDimensions);
    W_block *= cell_volume / delta_t;     // W_block = Mass matrix / delta_t

    // Create Upscaler and Solve
    smoothg::Graph graph(vertex_edge, edge_d_td, partitioning, weight, W_block);
    smoothg::GraphUpscale upscale(graph, {spect_tol, max_evects, hybridization, num_levels, ml_factor, elim_edges});

    upscale.PrintInfo();
    upscale.ShowSetupTime();

    // Input Vectors
    smoothg::Vector fine_rhs = upscale.GetVector(0);
    fine_rhs = 0.0;
    //smoothg::Vector fine_rhs = VectorToVector(rhs_u_fine);
    //fine_rhs /= -10.0;

    smoothg::Vector fine_u = InitialCondition(ufespace, initial_val);
    smoothg::Vector fine_tmp(fine_u.size());

    std::vector<smoothg::Vector> ml_tmp = upscale.GetMLVectors();
    std::vector<smoothg::Vector> ml_work_u = upscale.GetMLVectors();
    std::vector<smoothg::Vector> ml_work_rhs = upscale.GetMLVectors();

    for (int i = 0; i < num_levels; ++i)
    {
        upscale.Restrict(fine_u, ml_work_u[i]);
        upscale.Restrict(fine_rhs, ml_work_rhs[i]);
    }

    // Setup visualization
    mfem::ParGridFunction field(&ufespace);

    std::vector<mfem::socketstream> ml_vis_v(num_levels);

    if (vis_step > 0)
    {
        bool show_log = false;
        VisRange vis_range{-initial_val, initial_val};
        VisType vis_type = VisType::Angle;

        std::string title = "pressure";
        VectorToField(fine_u, field);

        for (int i = 0; i < num_levels; ++i)
        {
            std::string caption = "Level " + std::to_string(i);
            VisSetup(comm, ml_vis_v[i], field, *pmesh, vis_range, title, caption, show_log, vis_type);
            MPI_Barrier(comm);
        }
    }

    // Time Stepping

    double time = 0.0;
    int count = 0;

    Timer chrono(Timer::Start::True);

    while (time < total_time)
    {
        for (int i = 0; i < num_levels; ++i)
        {
            upscale.GetMatrix(i).LocalW().Mult(ml_work_u[i], ml_tmp[i]);

            ml_tmp[i] += ml_work_rhs[i];
            ml_tmp[i] *= -1.0;

            upscale.SolveLevel(i, ml_tmp[i], ml_work_u[i]);
            MPI_Barrier(comm);
        }

        if (myid == 0)
        {
            std::cout << std::fixed << std::setw(8) << count << "\t" << time << "\n";
        }

        time += delta_t;
        count++;

        if (vis_step > 0 && count % vis_step == 0)
        {
            upscale.Interpolate(ml_work_u[0], fine_tmp);
            upscale.Orthogonalize(0, fine_tmp);

            VectorToField(fine_tmp, field);
            VisUpdate(comm, ml_vis_v[0], field, *pmesh);
            MPI_Barrier(comm);

            for (int i = 1; i < num_levels; ++i)
            {
                upscale.Interpolate(ml_work_u[i], fine_u);
                upscale.Orthogonalize(0, fine_u);
                MPI_Barrier(comm);

                VectorToField(fine_u, field);
                VisUpdate(comm, ml_vis_v[i], field, *pmesh);
                MPI_Barrier(comm);

                auto error = smoothg::CompareError(comm, fine_u, fine_tmp) * 100; // as percent
                ParPrint(myid, std::cout << "\t Level " << i << " Pressure Error " << error << "%\n");
                MPI_Barrier(comm);
            }
        }
    }

    chrono.Click();

    if (myid == 0)
    {
        std::cout << "Total Time: " << chrono.TotalTime() << "\n";
    }

    return 0;
}

smoothg::Vector InitialCondition(mfem::ParFiniteElementSpace& ufespace, double initial_val)
{
    HalfCoeffecient half(initial_val);

    mfem::GridFunction init(&ufespace);
    init.ProjectCoefficient(half);

    int size = init.Size();
    smoothg::Vector vect(init.Size());

    for (int i = 0; i < size; ++i)
    {
        vect[i] = init[i];
    }

    return vect;
}
