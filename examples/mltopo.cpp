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

#include "smoothG.hpp"

using namespace smoothg;

using linalgcpp::ReadText;
using linalgcpp::WriteText;
using linalgcpp::ReadCSR;

int main(int argc, char* argv[])
{
    // Initialize MPI
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // program options from command line
    std::string graph_filename = "../../graphdata/vertex_edge_sample.txt";
    std::string partition_filename = "../../graphdata/partition_sample.txt";
    bool save_output = false;

    int num_levels = 2;
    int coarsen_factor = 10;
    bool metis_agglomeration = false;

    bool generate_graph = false;
    int gen_vertices = 1000;
    int mean_degree = 40;
    double beta = 0.15;
    int seed = 0;

    linalgcpp::ArgParser arg_parser(argc, argv);

    arg_parser.Parse(graph_filename, "-g", "Graph connection data.");
    arg_parser.Parse(partition_filename, "-p", "Partition data.");
    arg_parser.Parse(save_output, "-save", "Save topology partitionings.");
    arg_parser.Parse(num_levels, "-nl", "Number of topology levels.");
    arg_parser.Parse(coarsen_factor, "-cf", "Coarsening factor between levels.");
    arg_parser.Parse(metis_agglomeration, "-ma", "Enable Metis partitioning for initial partition.");
    arg_parser.Parse(generate_graph, "-gg", "Generate a graph.");
    arg_parser.Parse(gen_vertices, "-nv", "Number of vertices of generated graph.");
    arg_parser.Parse(mean_degree, "-md", "Average vertex degree of generated graph.");
    arg_parser.Parse(beta, "-b", "Probability of rewiring in the Watts-Strogatz model.");
    arg_parser.Parse(seed, "-s", "Seed for random number generator.");

    if (!arg_parser.IsGood())
    {
        ParPrint(myid, arg_parser.ShowHelp());
        ParPrint(myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(myid, arg_parser.ShowOptions());

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

    std::vector<int> global_partitioning;
    if (metis_agglomeration || generate_graph)
    {
        double ubal = 1.0;
        global_partitioning = PartitionAAT(vertex_edge_global, coarsen_factor, ubal);
    }
    else
    {
        global_partitioning = ReadText<int>(partition_filename);
    }

    std::vector<GraphTopology> topos;

    Graph fine_graph(comm, vertex_edge_global, global_partitioning);
    topos.emplace_back(fine_graph);

    for (int i = 1; i < num_levels; ++i)
    {
        topos.emplace_back(topos.back(), coarsen_factor);
    }

    if (save_output)
    {
        auto Output = [&](int level, const std::vector<int>& part)
        {
            std::stringstream ss;
            ss << "topo_part_" << std::setw(5) << std::setfill('0') << level << ".txt";

            WriteVector(comm, part, ss.str(),
                        nvertices_global, fine_graph.vertex_map_);
        };

        SparseMatrix vertex_agg = topos[0].agg_vertex_local_.Transpose();

        Output(0, vertex_agg.GetIndices());

        for (int i = 1; i < topos.size(); ++i)
        {
            SparseMatrix vertex_agg_i = topos[i].agg_vertex_local_.Transpose();
            vertex_agg = vertex_agg.Mult(vertex_agg_i);

            Output(i, vertex_agg.GetIndices());
        }
    }

    return 0;
}
