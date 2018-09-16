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

#include <fstream>
#include <memory>

#include "spe10.hpp"
#include "Sampler.hpp"
#include "DarcySolver.hpp"

using namespace rs2000;
using namespace mfem;

// Computes various statistics and realizations of the SPDE sampler
// without mesh embedding (the variance will be artificially inflated
// along the boundary, especially near corners) solving the saddle point linear
// system with solver specified in XML parameter list.

int main (int argc, char* argv[])
{
    // Initialize MPI
    smoothg::MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    bool coarse_sample = false;

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
                   "Number of coarsening levels");
    int num_samples = 0;
    args.AddOption(&num_samples, "-ns", "--num-samples",
                   "Number of samples to generate");
    double corlen = 100;
    args.AddOption(&corlen, "-cor", "--corlen",
                   "Corelation length");
    int initial_seed = 1;
    args.AddOption(&initial_seed, "-is", "--initial-seed",
                   "Initial seed for sampelr");

    bool verbose = false;
    args.AddOption(&verbose, "-verb", "--verbose", "-no-verb", "--no-verbose",
                   "Verbose output.");
    // Uncertainty parameters
    bool lognormal = true;
    args.AddOption(&lognormal, "-lm", "--log-normal", "-no-lm", "--no-log-normal",
                   "Log Normal");
    int sample_level = 0;
    args.AddOption(&sample_level, "-sl", "--sample-level",
                   "Level to sample from");

    args.Parse();
    if (!args.Good())
    {
        ParPrint(myid, args.PrintUsage(std::cout));

        return EXIT_FAILURE;
    }

    ParPrint(myid, args.PrintOptions(std::cout));

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

    ParPrint(myid, std::cout << pmesh->GetNEdges() << " fine edges, " <<
             pmesh->GetNFaces() << " fine faces, " <<
             pmesh->GetNE() << " fine elements\n");

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

    // GridFunctions for visualization
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

    std::vector<int> elim_edges;
    int num_bdr = boundary_att_edge.Rows();

    for (int i = 0; i < num_bdr; ++i)
    {
        std::vector<int> edges = boundary_att_edge.GetIndices(i);
        elim_edges.insert(std::end(elim_edges), std::begin(edges),
                          std::end(edges));
    }

    std::vector<double> one_weight(mfem_weight.Size(), 1.0);

    for (auto&& edge : elim_edges)
    {
        mfem_weight[edge] *= 2.0;
        one_weight[edge] *= 2.0;
    }

    std::vector<double> weight(mfem_weight.Size());
    for (int i = 0; i < mfem_weight.Size(); ++i)
    {
        weight[i] = 1.0 / mfem_weight[i];
        one_weight[i] = 1.0 / one_weight[i];
    }

    // Create Upscaler and Solve

    double kappa = 1.0 / corlen;

    smoothg::SparseMatrix W_block = smoothg::SparseIdentity(vertex_edge.Rows());
    //double cell_volume = spe10problem.CellVolume(nDimensions) * (num_refine + 1);
    double cell_volume = spe10problem.CellVolume(nDimensions);
    W_block *= cell_volume * kappa * kappa;

    // Set up GraphUpscale
    /// [Upscale]
    double coarsen_factor = 4.0;
    smoothg::UpscaleParams params(spect_tol, max_evects, hybridization,
                                  num_levels, coarsen_factor, elim_edges);

    smoothg::Graph sampler_graph(vertex_edge, edge_d_td, partitioning, one_weight, W_block);
    smoothg::Graph graph(vertex_edge, edge_d_td, partitioning, weight);

    smoothg::GraphUpscale upscale(graph, params);

    rs2000::PDESampler sampler(std::move(sampler_graph), params,
                               nDimensions, corlen, cell_volume, lognormal);

    sampler.SetFESpace(&ufespace);

    auto rhs = upscale.GetBlockVector(0);
    rhs.GetBlock(0) = 0.0;
    rhs.GetBlock(1) = VectorToVector(rhs_u_fine);
    rhs.GetBlock(1) /= linalgcpp::ParL2Norm(comm, rhs.GetBlock(1));

    rs2000::DarcySolver solver(upscale);
    solver.SetRHS(rhs);

    Vector xi, coef;
    smoothg::Vector upscaled_coeff;

    double Q;
    double C;

    std::vector<double> sample_times(num_levels, 0.0);
    std::vector<double> solve_times(num_levels, 0.0);

    for (int sample = 0; sample < num_samples; ++sample)
    {
        sampler.Sample(sample_level, xi);

        if (visualization)
        {
            auto fine_xi = sampler.GetUpscale().Interpolate(VectorToVector(xi));
            auto range = GetVisRange(comm, fine_xi);
            Visualize(VectorToVector(fine_xi), *pmesh, vertex_gf, range,
                      "pde xi - Sample: " + std::to_string(sample),
                      sample_level, lognormal);
        }

        for (int ilevel = sample_level; ilevel < num_levels; ilevel++)
        {
            sampler.Eval(ilevel, xi, coef);
            solver.SolveFwd(ilevel, coef, Q, C);

            sample_times[ilevel] += sampler.SampleTime(ilevel);
            solve_times[ilevel] += solver.GetSolveTime(ilevel);

            if (visualization)
            {
                // Vis Solution
                {
                    auto fine_sol = upscale.Interpolate(solver.Solution(ilevel), 0);

                    auto sol_range = GetVisRange(comm, fine_sol);
                    Visualize(VectorToVector(fine_sol.GetBlock(1)), *pmesh, vertex_gf,
                              sol_range, "pde sol - Sample: " + std::to_string(sample),
                              ilevel, lognormal);
                }

                // Vis Coefficients
                {
                    solver.InterpolateCoeff(ilevel, upscaled_coeff);

                    auto coeff_range = GetVisRange(comm, upscaled_coeff);
                    Visualize(VectorToVector(upscaled_coeff), *pmesh, vertex_gf,
                              coeff_range, "coefficients - Sample: " + std::to_string(sample),
                              ilevel, lognormal);
                }
            }
        }
    }

    if (myid == 0)
    {
        std::cout << std::string(50, '-') << "\n";
        std::cout << "Sample Time:\n";

        std::cout << "Level\tTotal\tAvg\n";
        for (int i = sample_level; i < num_levels; ++i)
        {
            std::cout << std::setprecision(4) << std::fixed
                      << i << "\t" << sample_times[i] << "\t"
                      << sample_times[i] / num_samples << "\n";
        }

        std::cout << std::string(50, '-') << "\n";
        std::cout << "Solver Time:\n";

        std::cout << "Level\tTotal\tAvg\n";
        for (int i = sample_level; i < num_levels; ++i)
        {
            std::cout << std::setprecision(4) << std::fixed
                      << i << "\t" << solve_times[i] << "\t"
                      << solve_times[i] / num_samples << "\n";
        }

        std::cout << std::string(50, '-') << "\n";
    }

    return EXIT_SUCCESS;
}

