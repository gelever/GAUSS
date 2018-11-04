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

#include "Sampler.hpp"
#include "spe10.hpp"

using namespace rs2001;
using namespace mfem;

double dot(const Vector& a, const Vector& b, MPI_Comm comm);
HypreParVector* chi_center_of_mass(ParMesh* pmesh);

template <int coord>
inline double xfun(mfem::Vector& v)
{
    return v(coord);
}

// Computes various statistics and realizations of the SPDE sampler
// without mesh embedding (the variance will be artificially inflated
// along the boundary, especially near corners) solving the saddle point linear
// system with solver specified in XML parameter list.

int main (int argc, char* argv[])
{
    // Initialize MPI
    gauss::MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm;
    int myid = mpi_info.myid;
    int num_procs = mpi_info.num_procs;

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
    int num_samples = 3;
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
    gauss::SparseMatrix vertex_edge = TableToSparse(vertex_edge_table);

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

    gauss::ParMatrix edge_d_td = ParMatrixToParMatrix(*sigmafespace.Dof_TrueDof_Matrix());
    gauss::SparseMatrix edge_boundary_att = GenerateBoundaryAttributeTable(*pmesh);
    gauss::SparseMatrix boundary_att_edge = edge_boundary_att.Transpose();

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
        one_weight[edge] /= 2.0;
    }

    std::vector<double> weight(mfem_weight.Size());
    for (int i = 0; i < mfem_weight.Size(); ++i)
    {
        weight[i] = 1.0 / mfem_weight[i];
        //one_weight[i] = 1.0 / one_weight[i];
    }

    ///*
    gauss::Vector vect(weight);
    VisRange range = GetVisRange(comm, vect);

    Visualize(VectorToVector(vect), *pmesh, edge_gf,
              range, "Initial Weight", 0, true);
    //*/

    // Create Upscaler and Solve

    double kappa = 1.0 / corlen;

    gauss::SparseMatrix W_block = gauss::SparseIdentity(vertex_edge.Rows());
    //double cell_volume = spe10problem.CellVolume(nDimensions) * (num_refine + 1);
    double cell_volume = spe10problem.CellVolume(nDimensions) / (num_refine + 1);
    W_block *= cell_volume * kappa * kappa;

    // Set up GraphUpscale
    /// [Upscale]
    double coarsen_factor = 4.0;

    gauss::Graph sampler_graph(vertex_edge, edge_d_td, partitioning, one_weight, W_block);
    gauss::UpscaleParams params(spect_tol, max_evects, hybridization,
                                num_levels, coarsen_factor, elim_edges);

    rs2001::PDESampler sampler(std::move(sampler_graph), params,
                               nDimensions, corlen, cell_volume, lognormal);
    sampler.SetFESpace(&ufespace);

    auto h_chi = std::unique_ptr<HypreParVector>(chi_center_of_mass(pmesh));

    std::vector<mfem::Vector> chi(sampler.GetUpscale().NumLevels());

    chi[0] = *h_chi;

    for (int i = 1; i < num_levels; ++i)
    {
        chi[i] = sampler.Restrict(i, chi[0]);
    }

    Vector xi, coef;
    sampler.Sample(0, xi);

    VisRange realize_range;
    for (int ilevel = 0; ilevel < num_levels; ilevel++)
    {
        sampler.Eval(ilevel, xi, coef);

        if (visualization)
        {
            {
                auto vect = sampler.Interpolate(ilevel, coef);
                if (ilevel == 0)
                {
                    realize_range = GetVisRange(comm, vect);
                }

                Visualize(VectorToVector(vect), *pmesh, vertex_gf,
                          realize_range, "pde realization", ilevel, lognormal);
            }
        }
    }

    int nsamples = num_samples;
    int nLevels = num_levels;
    double variance = 1.0;
    bool visualize = visualization;
    bool print_time = true;

    Vector exp_error(nLevels);
    Vector var_error(nLevels);
    const double exact_expectation(lognormal ? std::exp(variance / 2.) : 0.0);
    const double exact_variance(lognormal ?
                                std::exp(variance) * (std::exp(variance) - 1.) : variance);
    // Dof stats
    Vector nnz(nLevels);
    Vector ndofs_l(nLevels);
    Vector ndofs_g(nLevels);
    Vector stoch_size_l(nLevels);
    Vector stoch_size_g(nLevels);

    std::vector<mfem::Vector> expectation(nLevels);
    std::vector<mfem::Vector> chi_cov(nLevels);
    std::vector<mfem::Vector> marginal_variance(nLevels);

    std::vector<gauss::Vector> vis_xi;
    std::vector<int> vis_xi_level;
    std::vector<gauss::Vector> vis_sol;
    std::vector<int> vis_sol_level;
    bool vis_every_sample = true && num_samples < 10;


    for (int ilevel(0); ilevel < nLevels; ++ilevel)
    {
        int s_size = sampler.SampleSize(ilevel);

        expectation[ilevel].SetSize(s_size);
        chi_cov[ilevel].SetSize(s_size);
        marginal_variance[ilevel].SetSize(s_size);

        expectation[ilevel] = 0.0;
        chi_cov[ilevel] = 0.0;
        marginal_variance[ilevel] = 0.0;

        {
            if (!myid && verbose) std::cout << "Level " << ilevel << ":\n";

            for (int i(0); i < nsamples; ++i)
            {
                sampler.Sample(ilevel, xi);

                if (vis_every_sample)
                {
                    vis_xi.push_back(sampler.Interpolate(ilevel, xi));
                    vis_xi_level.push_back(ilevel);
                }

                sampler.Eval(ilevel, xi, coef);

                if (vis_every_sample)
                {
                    vis_sol.push_back(sampler.Interpolate(ilevel, coef));
                    vis_sol_level.push_back(ilevel);
                }

                double chi_coef = dot(coef, chi[ilevel], comm);
                chi_cov[ilevel].Add(chi_coef, coef);

                expectation[ilevel].Add(1., coef);

                for (int k = 0; k < s_size; ++k)
                {
                    marginal_variance[ilevel](k) += coef(k) * coef(k);
                }
            }
        }

        nnz[ilevel] = sampler.GetNNZ(ilevel);
        ndofs_l[ilevel] = sampler.SampleSize(ilevel);
        ndofs_g[ilevel] = sampler.GlobalSampleSize(ilevel);
        stoch_size_l[ilevel] = sampler.GetNumberOfDofs(ilevel);
        stoch_size_g[ilevel] = sampler.GetGlobalNumberOfDofs(ilevel);

        chi_cov[ilevel] *= 1. / static_cast<double>(nsamples);
        expectation[ilevel] *= 1. / static_cast<double>(nsamples);
        marginal_variance[ilevel] *= 1. / static_cast<double>(nsamples);

        // Error calculation
        exp_error[ilevel] = sampler.ComputeL2Error(
                                ilevel, expectation[ilevel], exact_expectation);
        var_error[ilevel] = sampler.ComputeL2Error(
                                ilevel, marginal_variance[ilevel], exact_variance);

    }

    for (int i = 0; i < vis_xi.size(); ++i)
    {
        VisRange vis_range = GetVisRange(comm, vis_xi[0]);
        Visualize(VectorToVector(vis_xi[i]), *pmesh, vertex_gf,
                  vis_range, "Sample Xi", vis_xi_level[i], lognormal);

    }

    for (int i = 0; i < vis_sol.size(); ++i)
    {
        VisRange vis_range = GetVisRange(comm, vis_sol[i]);
        Visualize(VectorToVector(vis_sol[i]), *pmesh, vertex_gf,
                  vis_range, "Sample Sol", vis_sol_level[i], lognormal);

    }

    if (visualization)
    {

        for (int ilevel(0); ilevel < nLevels; ++ilevel)
        {
            {
                auto vect = sampler.Interpolate(ilevel, (expectation[ilevel]));
                VisRange range = GetVisRange(comm, vect);

                Visualize(VectorToVector(vect), *pmesh, vertex_gf,
                          range, "pde expectation", ilevel, lognormal);
            }
            {
                auto vect = sampler.Interpolate(ilevel, (chi_cov[ilevel]));
                VisRange range = GetVisRange(comm, vect);

                Visualize(VectorToVector(vect), *pmesh, vertex_gf,
                          range, "pde cov chi", ilevel, lognormal);
            }
            {
                auto vect = sampler.Interpolate(ilevel, (marginal_variance[ilevel]));
                VisRange range = GetVisRange(comm, vect);

                Visualize(VectorToVector(vect), *pmesh, vertex_gf,
                          range, "pde marginal variance", ilevel, lognormal);
            }
        }
    }


    if (myid == 0)
    {
        std::cout << "\nSampler Error: Expected E[u] = " << exact_expectation
                  << ",  Expected V[u] = " << exact_variance << '\n'
                  << "\n L2 Error PDE Sampler \n";
    }

    return EXIT_SUCCESS;
}

HypreParVector* chi_center_of_mass(ParMesh* pmesh)
{
    const int nDimensions = pmesh->Dimension();
    FiniteElementCollection* fec = new L2_FECollection(0, nDimensions);
    ParFiniteElementSpace* fes = new ParFiniteElementSpace(pmesh, fec);

    ConstantCoefficient one_coeff(1.);
    Array< FunctionCoefficient* > xcoeffs(nDimensions);
    xcoeffs[0] = new FunctionCoefficient(xfun<0>);
    xcoeffs[1] = new FunctionCoefficient(xfun<1>);
    if (nDimensions == 3)
        xcoeffs[2] = new FunctionCoefficient(xfun<2>);

    ParGridFunction ones(fes);
    ones.ProjectCoefficient(one_coeff);
    HypreParVector* ones_v = ones.GetTrueDofs();

    ParLinearForm average(fes);
    average.AddDomainIntegrator(new DomainLFIntegrator(one_coeff));
    average.Assemble();
    HypreParVector* average_v = average.ParallelAssemble();

    Array< ParGridFunction* > xi(nDimensions);
    Array< HypreParVector* > xi_v(nDimensions);
    const double volume = InnerProduct(average_v, ones_v);
    Vector cm(nDimensions);
    for (int i = 0; i < nDimensions; ++i)
    {
        xi[i] = new ParGridFunction(fes);
        xi[i]->ProjectCoefficient(*xcoeffs[i]);
        xi_v[i] = xi[i]->GetTrueDofs();
        cm(i) = InnerProduct(average_v, xi_v[i]) / volume;
    }
    const int ne = ones_v->Size();
    double minimum = 1e10;
    int minid = -1;
    for (int i = 0; i < ne; ++i)
    {
        (*ones_v)(i) = 0.0;
        for (int idim = 0; idim < nDimensions; ++idim)
        {
            double d = cm(idim) - (*xi_v[idim])(i);
            (*ones_v)(i) += d * d;
        }
        (*ones_v)(i) = sqrt((*ones_v)(i));
        if ((*ones_v)(i) < minimum)
        {
            minimum = (*ones_v)(i);
            minid = i;
        }
    }
    double gmin = minimum;
    MPI_Allreduce(&minimum, &gmin, 1, MPI_DOUBLE, MPI_MIN, pmesh->GetComm());

    (*ones_v) = 0.0;
    if (minimum == gmin)
        (*ones_v)(minid) = 1.;

    for (int i = 0; i < nDimensions; ++i)
    {
        delete xi[i];
        delete xi_v[i];
        delete xcoeffs[i];
    }
    delete average_v;
    delete fes;
    delete fec;

    return ones_v;
}

double dot(const Vector& a,
           const Vector& b,
           MPI_Comm comm)
{
    double ldot = a * b;
    double gdot = 0.;
    MPI_Allreduce(&ldot, &gdot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return gdot;
}
