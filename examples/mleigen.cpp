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

/** @file mleigen.cpp
    @brief Example usage of multilevel eigensolver
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "smoothG.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::WriteText;
using linalgcpp::ReadCSR;

using parlinalgcpp::ParOperator;
using parlinalgcpp::LOBPCG;
using parlinalgcpp::ParCG;
using parlinalgcpp::BoomerAMG;

/// @brief Computes DMinvDt + shift * I
class ShiftedDMinvDt : public ParOperator
{
    public:
        /** @brief Construtor
            @param M M matrix
            @param D D matrix
            @param shift shift to apply
        */
        ShiftedDMinvDt(const ParMatrix& M, const ParMatrix& D, double shift = 1.0)
            : ParOperator(D.GetComm(), D.GetRowStarts()),
              M_prec_(M, 1, 1e-8), M_solver_(M, M_prec_, 5000 /* max_iter */ , 1e-8 /* tol */),
              D_(D), DTx_(D_.Cols()), MinvDTx_(D_.Cols()),
              shift_(shift) { }

        /** @brief Compute y = (DMinvD^T + shift * I)x
            @param input input vector x
            @param output output vector y
        */
        void Mult(const VectorView& input, VectorView output) const
        {
            D_.MultAT(input, DTx_);
            M_solver_.Mult(DTx_, MinvDTx_);
            D_.Mult(MinvDTx_, output);

            output.Add(shift_, input);
        }

    private:
        BoomerAMG M_prec_;
        ParCG M_solver_;
        ParMatrix D_;

        mutable Vector DTx_;
        mutable Vector MinvDTx_;

        double shift_;
};

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    std::string graph_filename = "../../graphdata/vertex_edge_sample.txt";

    double coarse_factor = 100;
    int max_evects = 5;
    double spect_tol = 1.0;
    bool hybridization = true;

    double shift = 1.0;
    int num_modes = 4;
    bool no_coarse = false;
    bool verbose = false;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(graph_filename, "--g", "Graph connection data.");
    arg_parser.Parse(coarse_factor, "--coarse-factor", "Coarsening factor for partition.");
    arg_parser.Parse(max_evects, "--m", "Maximum eigenvectors per aggregate.");
    arg_parser.Parse(spect_tol, "--t", "Spectral tolerance for eigenvalue problem.");
    arg_parser.Parse(hybridization, "--hb", "Use hybridization in coarse solver.");
    arg_parser.Parse(num_modes, "--num-modes", "Number of eigenpairs to compute.");
    arg_parser.Parse(no_coarse, "--no-coarse", "Do not use coarse approximation as initial guess.");
    arg_parser.Parse(verbose, "--verbose", "Verbose output.");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

    SparseMatrix vertex_edge = ReadCSR(graph_filename);

    int num_vertices_global = vertex_edge.Rows();
    int num_edges_global = vertex_edge.Cols();

    ParPrint(myid, std::cout << "\nUpscaling:" << std::endl);

    GraphUpscale upscale(comm, vertex_edge, coarse_factor,
                         spect_tol, max_evects, hybridization);

    upscale.ShowSetupTime();
    ParPrint(myid, std::cout << "\nEigensolving:" << std::endl);

    Timer timer(Timer::Start::True);

    // Two level spectral graph Laplacian eigensolver
    std::vector<Vector> evects(num_modes, Vector(upscale.Rows()));

    if (no_coarse)
    {
        for (auto& evect : evects)
        {
            Randomize(evect);
            Normalize(evect);
        }
    }
    else
    {
        UpscaleCoarseSolve upscale_coarse(upscale);
        upscale.AssembleCoarseM();

        ShiftedDMinvDt A_c(upscale.GetCoarseMatrix().GlobalM(),
                           upscale.GetCoarseMatrix().GlobalD(),
                           shift);

        std::vector<Vector> evects_c(num_modes, Vector(upscale_coarse.Rows()));

        for (auto& evect : evects_c)
        {
            Randomize(evect);
            Normalize(evect);
        }

        auto evals = LOBPCG(A_c, evects_c, &upscale_coarse, verbose);

        for (auto eval : evals)
        {
            ParPrint(myid, std::cout << "Coarse Eval: " << (eval - shift) << "\n");
        }

        for (int i = 0; i < num_modes; ++i)
        {
            upscale.Interpolate(evects_c[i], evects[i]);
        }
    }

    ParMatrix A = upscale.GetFineMatrix().ToPrimal();
    A.AddDiag(shift);

    BoomerAMG boomer(A);

    auto evals = LOBPCG(A, evects, &boomer, verbose);

    timer.Click();

    for (auto eval : evals)
    {
        ParPrint(myid, std::cout << "Fine Eval: " << (eval - shift) << "\n");
    }

    ParPrint(myid, std::cout << "\nEigen Solve Time: " << timer.TotalTime() << "\n");

    return EXIT_SUCCESS;
}
