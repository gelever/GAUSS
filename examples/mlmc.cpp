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
   @file mlmc.cpp
   @brief This is an example for upscaling a graph Laplacian,
   where we change coefficients in the model without re-coarsening.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>
#include <random>

#include "smoothG.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::WriteText;
using linalgcpp::ReadCSR;

using parlinalgcpp::LOBPCG;
using parlinalgcpp::BoomerAMG;

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts);
Vector ComputeFiedlerVector(const MixedMatrix& mgl);

class NormalDistribution
{
    public:
        NormalDistribution(double mean = 0.0, double stddev = 1.0, int seed = 0)
            : generator_(seed), dist_(mean, stddev) { }

        ~NormalDistribution() = default;

        double Sample() { return dist_(generator_); }

    private:
        std::mt19937 generator_;
        std::normal_distribution<double> dist_;
};

class SamplerUpscale
{
    public:

        SamplerUpscale(Graph graph, double spect_tol, int max_evects, bool hybridization,
                       int dimension, double kappa, double cell_volume, int seed);

        ~SamplerUpscale() = default;

        void Sample();

        const std::vector<double>& GetCoefficientFine() const { return coefficient_fine_; }
        const std::vector<double>& GetCoefficientCoarse() const { return coeffecient_coarse_; }

    private:
        GraphUpscale upscale_;

        NormalDistribution normal_dist_;
        double cell_volume_;
        double scalar_g_;

        Vector rhs_fine_;
        Vector rhs_coarse_;

        Vector sol_fine_;
        Vector sol_coarse_;

        std::vector<double> coefficient_fine_;
        std::vector<double> coeffecient_coarse_;

        Vector constant_coarse_;
};


SamplerUpscale::SamplerUpscale(Graph graph, double spect_tol, int max_evects, bool hybridization,
                               int dimension, double kappa, double cell_volume, int seed)
    : upscale_(std::move(graph), spect_tol, max_evects, hybridization),
      normal_dist_(0.0, 1.0, seed),
      cell_volume_(cell_volume),
      rhs_fine_(upscale_.GetFineVector()),
      rhs_coarse_(upscale_.GetCoarseVector()),
      sol_fine_(upscale_.GetFineVector()),
      sol_coarse_(upscale_.GetCoarseVector()),
      coefficient_fine_(upscale_.Rows()),
      coeffecient_coarse_(upscale_.NumAggs()),
      constant_coarse_(upscale_.GetCoarseVector())
{
    upscale_.PrintInfo();
    upscale_.ShowSetupTime();

    Vector ones = upscale_.GetFineVector();
    ones = 1.0;

    upscale_.Restrict(ones, constant_coarse_);

    double nu_param = dimension == 2 ? 1.0 : 0.5;
    double ddim = static_cast<double>(dimension);

    scalar_g_ = std::pow(4.0 * M_PI, ddim / 4.0) * std::pow(kappa, nu_param) *
            std::sqrt( std::tgamma(nu_param + ddim / 2.0) / std::tgamma(nu_param) );
}

void SamplerUpscale::Sample()
{
    double g_cell_vol_sqrt = scalar_g_ * std::sqrt(cell_volume_);

    for (auto& i : rhs_fine_)
    {
        i = g_cell_vol_sqrt * normal_dist_.Sample();
    }

    // Set Fine Coeffecient
    upscale_.SolveFine(rhs_fine_, sol_fine_);

    int fine_size = sol_fine_.size();
    assert(coefficient_fine_.size() == fine_size);

    for (int i = 0; i < fine_size; ++i)
    {
        coefficient_fine_[i] = std::exp(sol_fine_[i]);
    }

    // Set Coarse Coeffecient
    upscale_.Restrict(rhs_fine_, rhs_coarse_);
    upscale_.SolveCoarse(rhs_coarse_, sol_coarse_);

    std::fill(std::begin(coeffecient_coarse_), std::end(coeffecient_coarse_), 0.0);

    int coarse_size = sol_coarse_.size();
    int agg_index = 0;

    assert(constant_coarse_.size() == coarse_size);

    for (int i = 0; i < coarse_size; ++i)
    {
        if (std::fabs(constant_coarse_[i]) > 1e-8)
        {
            coeffecient_coarse_[agg_index++] = std::exp(sol_coarse_[i] / constant_coarse_[i]);
        }
    }

    assert(agg_index == upscale_.NumAggs());
}

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    std::string graph_filename = "../../graphdata/fe_vertex_edge.txt";
    std::string fiedler_filename = "../../graphdata/fe_rhs.txt";
    std::string partition_filename = "../../graphdata/fe_part.txt";
    std::string weight_filename = "../../graphdata/fe_weight_0.txt";
    std::string w_block_filename = "";
    bool save_output = false;

    int isolate = -1;
    int max_evects = 4;
    double spect_tol = 1e-3;
    int num_partitions = 12;
    bool hybridization = false;
    bool metis_agglomeration = false;

    bool generate_fiedler = false;
    bool save_fiedler = false;

    bool generate_graph = false;
    int gen_vertices = 1000;
    int mean_degree = 40;
    double beta = 0.15;
    int seed = 0;

    int num_samples = 3;
    int dimension = 2;
    double kappa = 0.001;
    double cell_volume = 200.0;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(graph_filename, "--g", "Graph connection data.");
    arg_parser.Parse(fiedler_filename, "--f", "Fiedler vector data.");
    arg_parser.Parse(partition_filename, "--p", "Partition data.");
    arg_parser.Parse(weight_filename, "--w", "Edge weight data.");
    arg_parser.Parse(w_block_filename, "--wb", "W block data.");
    arg_parser.Parse(save_output, "--save", "Save solutions.");
    arg_parser.Parse(isolate, "--isolate", "Isolate a single vertex.");
    arg_parser.Parse(max_evects, "--m", "Maximum eigenvectors per aggregate.");
    arg_parser.Parse(spect_tol, "--t", "Spectral tolerance for eigenvalue problem.");
    arg_parser.Parse(num_partitions, "--np", "Number of partitions to generate.");
    arg_parser.Parse(hybridization, "--hb", "Enable hybridization.");
    arg_parser.Parse(metis_agglomeration, "--ma", "Enable Metis partitioning.");
    arg_parser.Parse(generate_fiedler, "--gf", "Generate Fiedler vector.");
    arg_parser.Parse(save_fiedler, "--sf", "Save a generated Fiedler vector.");
    arg_parser.Parse(generate_graph, "--gg", "Generate a graph.");
    arg_parser.Parse(gen_vertices, "--nv", "Number of vertices of generated graph.");
    arg_parser.Parse(mean_degree, "--md", "Average vertex degree of generated graph.");
    arg_parser.Parse(beta, "--b", "Probability of rewiring in the Watts-Strogatz model.");
    arg_parser.Parse(seed, "--s", "Seed for random number generator.");
    arg_parser.Parse(num_samples, "--ns", "Number of samples.");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    /// [Load graph from file or generate one]
    SparseMatrix vertex_edge_global;

    if (generate_graph)
    {
        vertex_edge_global = GenerateGraph(comm, gen_vertices, mean_degree, beta, seed);
    }
    else
    {
        vertex_edge_global = ReadCSR(graph_filename);
    }

    const int nvertices_global = vertex_edge_global.Rows();
    const int nedges_global = vertex_edge_global.Cols();
    /// [Load graph from file or generate one]

    /// [Partitioning]
    std::vector<int> part;
    if (metis_agglomeration || generate_graph)
    {
        assert(num_partitions >= num_procs);
        part = MetisPart(vertex_edge_global, num_partitions);
    }
    else
    {
        part = ReadText<int>(partition_filename);
    }
    /// [Partitioning]

    /// [Load the edge weights]
    std::vector<double> weight;
    if (!weight_filename.empty())
    {
        weight = linalgcpp::ReadText(weight_filename);
    }
    else
    {
        weight = std::vector<double>(nedges_global, 1.0);
    }

    std::vector<double> one_weight(weight.size(), 1.0);

    SparseMatrix W_block = SparseIdentity(nvertices_global);
    W_block *= cell_volume * kappa * kappa;

    // Set up GraphUpscale
    /// [Upscale]
    Graph sampler_graph(comm, vertex_edge_global, part, one_weight, W_block);
    Graph graph(comm, vertex_edge_global, part, weight);

    int sampler_seed = myid + 1;
    SamplerUpscale sampler(std::move(sampler_graph), spect_tol, max_evects, hybridization,
                           dimension, kappa, cell_volume, sampler_seed);
    GraphUpscale upscale(std::move(graph), spect_tol, max_evects, hybridization);

    /// [Upscale]

    /// [Right Hand Side]
    BlockVector fine_rhs = upscale.GetFineBlockVector();
    fine_rhs.GetBlock(0) = 0.0;

    if (generate_graph || generate_fiedler)
    {
        fine_rhs.GetBlock(1) = ComputeFiedlerVector(upscale.GetFineMatrix());
    }
    else
    {
        fine_rhs.GetBlock(1) = upscale.ReadVertexVector(fiedler_filename);
    }
    /// [Right Hand Side]

    /// [Solve]

    BlockVector fine_sol = upscale.GetFineBlockVector();
    BlockVector upscaled_sol = upscale.GetFineBlockVector();

    for (int i = 0; i < num_samples; ++i)
    {
        sampler.Sample();

        const auto& fine_coeff = sampler.GetCoefficientFine();
        const auto& coarse_coeff = sampler.GetCoefficientCoarse();

        upscale.MakeCoarseSolver(coarse_coeff);
        upscale.MakeFineSolver(fine_coeff);

        upscale.Solve(fine_rhs, upscaled_sol);
        upscale.SolveFine(fine_rhs, fine_sol);
        upscale.Orthogonalize(fine_sol);

        if (save_output)
        {
            std::stringstream ss_coarse;
            std::stringstream ss_fine;
            std::stringstream ss_coeffecient;

            ss_coarse << "coarse_sol_" << i << ".txt";
            ss_fine << "fine_sol_" << i << ".txt";
            ss_coeffecient << "fine_coeff_" << i << ".txt";

            upscale.WriteVertexVector(upscaled_sol.GetBlock(1), ss_coarse.str());
            upscale.WriteVertexVector(fine_sol.GetBlock(1), ss_fine.str());
            upscale.WriteVertexVector(sampler.GetCoefficientFine(), ss_coeffecient.str());
        }

        upscale.ShowCoarseSolveInfo();
        upscale.ShowFineSolveInfo();

        /// [Check Error]
        upscale.ShowErrors(upscaled_sol, fine_sol);
        /// [Check Error]

    }
    /// [Solve]

    if (save_fiedler)
    {
        upscale.WriteVertexVector(fine_rhs.GetBlock(1), fiedler_filename);
    }

    return 0;
}

std::vector<int> MetisPart(const SparseMatrix& vertex_edge, int num_parts)
{
    SparseMatrix edge_vertex = vertex_edge.Transpose();
    SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);

    double ubal_tol = 2.0;

    return Partition(vertex_vertex, num_parts, ubal_tol);
}

Vector ComputeFiedlerVector(const MixedMatrix& mgl)
{
    ParMatrix A = mgl.ToPrimal();

    bool use_w = mgl.CheckW();

    if (!use_w)
    {
        A.AddDiag(1.0);
    }

    BoomerAMG boomer(A);

    int num_evects = 2;
    std::vector<Vector> evects(num_evects, Vector(A.Rows()));
    for (Vector& evect : evects)
    {
        evect.Randomize();
    }

    std::vector<double> evals = LOBPCG(A, evects, &boomer);

    assert(static_cast<int>(evals.size()) == num_evects);
    if (!use_w)
    {
        assert(std::fabs(evals[0] - 1.0) < 1e-8);
        assert(std::fabs(evals[1] - 1.0) > 1e-8);
    }

    return std::move(evects[1]);
}
