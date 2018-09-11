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
   @file direchlet.cpp
   @brief This is an example for applying Direchlet boundary conditions
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "spe10.hpp"

using namespace rs2000;

smoothg::SparseMatrix ComputeD(const smoothg::GraphUpscale& upscale, const smoothg::SparseMatrix& boundary_att_vertex,
                               const std::vector<int>& ess_bdr);

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
    bool dual_target = false;
    args.AddOption(&dual_target, "-dt", "--dual-target", "-no-dt",
                   "--no-dual-target", "Use dual graph Laplacian in trace generation.");
    bool scaled_dual = false;
    args.AddOption(&scaled_dual, "-sd", "--scaled-dual", "-no-sd",
                   "--no-scaled-dual", "Scale dual graph Laplacian by (inverse) edge weight.");
    bool energy_dual = false;
    args.AddOption(&energy_dual, "-ed", "--energy-dual", "-no-ed",
                   "--no-energy-dual", "Use energy matrix in trace generation.");
    bool visualization = false;
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization", "Enable visualization.");
    int num_refine = 0;
    args.AddOption(&num_refine, "-nr", "--num-refine",
                   "Number of mesh refinements.");
    double coarse_factor = 1.0;
    args.AddOption(&coarse_factor, "-cf", "--coarse-factor",
                   "Scale coarsening factor.");
    int num_levels = 2;
    args.AddOption(&num_levels, "-nl", "--num-levels",
                   "Number of levels.");
    double ml_factor = 4.0;
    args.AddOption(&ml_factor, "-mf", "--ml-factor",
                   "Multilevel aggregate coarsening factor.");
    int bndr_type = 3;
    args.AddOption(&bndr_type, "-bt", "--boundary-type",
                   "Boundary type");
    double ess_val = 1.0;
    args.AddOption(&ess_val, "-bv", "--boundary-value",
                   "Boundary value");
    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(std::cout);
        }

        return EXIT_FAILURE;
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

    mfem::ParGridFunction edge_gf(&sigmafespace);
    mfem::ParGridFunction vertex_gf(&ufespace);

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
    smoothg::SparseMatrix boundary_att_vertex = vertex_edge.Mult(edge_boundary_att).Transpose();

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

    // Create Upscaler and Solve
    smoothg::Graph graph(vertex_edge, edge_d_td, partitioning, weight);
    smoothg::GraphUpscale upscale(graph, {spect_tol, max_evects, hybridization, num_levels, ml_factor, elim_edges});

    upscale.ShowSetupTime();
    upscale.PrintInfo();

    smoothg::BlockVector rhs_fine = upscale.GetBlockVector(0);
    smoothg::BlockVector ess_conditions = upscale.GetBlockVector(0);

    rhs_fine = 0.0;
    ess_conditions = 0.0;

    std::vector<int> ess_bdr(boundary_att_vertex.Rows(), 0);
    std::vector<double> ess_vals(boundary_att_vertex.Rows(), 0);

    switch(bndr_type)
    {
        case 0: { ess_bdr[0] = 1; ess_bdr[2] = 1; ess_vals[2] = ess_val; break; }
        case 1: { ess_bdr[2] = 1; ess_bdr[0] = 1; ess_vals[0] = ess_val; break; }
        case 2: { ess_bdr[1] = 1; ess_bdr[3] = 1; ess_vals[3] = ess_val; break; }
        case 3: { ess_bdr[3] = 1; ess_bdr[1] = 1; ess_vals[1] = ess_val; break; }
        default: throw std::runtime_error("Invalid Boundary Type!");
    }


    for (int attr = 0; attr < ess_bdr.size(); ++attr)
    {
        if (ess_bdr[attr])
        {
            auto vertices = boundary_att_vertex.GetIndices(attr);

            for (auto&& dof : vertices)
            {
                ess_conditions.GetBlock(1)[dof] = ess_vals[attr];
                rhs_fine.GetBlock(1)[dof] = ess_vals[attr];
            }
        }
    }

    EliminateEssentialBC(upscale, boundary_att_vertex, ess_bdr, ess_conditions, rhs_fine);

    {
        VisRange range = GetVisRange(comm, ess_conditions.GetBlock(1));
        Visualize(VectorToVector(ess_conditions.GetBlock(1)), *pmesh, vertex_gf,
                range, "Ess U data", 0);
    }

    {
        VisRange range = GetVisRange(comm, rhs_fine.GetBlock(1));
        Visualize(VectorToVector(rhs_fine.GetBlock(1)), *pmesh, vertex_gf,
                  range, "Ess Bdr RHS", 0);
    }

    auto sols = upscale.MultMultiLevel(rhs_fine);
    upscale.ShowFineSolveInfo();
    upscale.ShowCoarseSolveInfo();

    const auto& M = upscale.GetMatrix(0).LocalM();
    auto D_interior = ComputeD(upscale, boundary_att_vertex, ess_bdr);

    for (int i = 1; i < sols.size(); ++i)
    {
        ParPrint(myid, std::cout << "Solution Level " << i << "\n");

        smoothg::Vector sigma_diff(sols[0].GetBlock(0));
        sigma_diff -= sols[i].GetBlock(0);

        auto info = smoothg::ComputeErrors(comm, M, D_interior, sols[i], sols[0]);
        info.back() = parlinalgcpp::ParL2Norm(comm, D_interior.Mult(sigma_diff));

        ParPrint(myid, smoothg::ShowErrors(info));
    }

    if (visualization)
    {
        bool show_log = false;
        VisType vis_type = VisType::Angle;
        VisRange vertex_range{0.0, ess_val};

        for (int level = 0; level < num_levels; ++level)
        {
            Visualize(VectorToVector(sols[level].GetBlock(1)), *pmesh, vertex_gf,
                      vertex_range, "Pressure", level, show_log, vis_type);
        }
    }

    return EXIT_SUCCESS;
}

smoothg::SparseMatrix ComputeD(const smoothg::GraphUpscale& upscale, const smoothg::SparseMatrix& boundary_att_vertex,
                               const std::vector<int>& ess_bdr)
{
    smoothg::SparseMatrix D = upscale.GetMatrix(0).LocalD();

    for (int i = 0; i < boundary_att_vertex.Rows(); ++i)
    {
        if (ess_bdr[i])
        {
            auto vertices = boundary_att_vertex.GetIndices(i);

            for (auto&& vertex : vertices)
            {
                D.EliminateRow(vertex);
            }
        }
    }

    return D;
}
