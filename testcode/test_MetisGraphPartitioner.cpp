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
   Test metis partitioning, just make sure it runs.
*/

#include <fstream>
#include <sstream>
#include <mpi.h>

#include "GAUSS.hpp"

using namespace gauss;
using linalgcpp::ReadCSR;
using linalgcpp::operator<<;

int main(int argc, char* argv[])
{
    // load the graph
    std::string graph_filename = "../../graphdata/vertex_edge_tiny.txt";
    SparseMatrix vertex_edge = ReadCSR(graph_filename);

    // partition
    int num_parts = 2;
    double ubal_tol = 2.0;

    SparseMatrix edge_vertex = vertex_edge.Transpose();
    SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);
    vertex_vertex.PrintDense("Vertex - Vertex");

    std::vector<int> partition = Partition(vertex_vertex, num_parts, ubal_tol);

    // check partition
    int size = partition.size();
    bool failed = false;

    for (int i = 0; i < size / 2; ++i)
    {
        std::cout << "partition[" << i << "] = " << partition[i] << "\n";

        if (partition[i] != 1)
        {
            failed = true;
        }
    }

    for (int i = size / 2; i < size; ++i)
    {
        std::cout << "partition[" << i << "] = " << partition[i] << "\n";

        if (partition[i] != 0)
        {
            failed = true;
        }
    }

    if (failed)
    {
        std::cerr << "Unexpected partitioning from metis!" << std::endl;
    }


    // Test PostIsolating

    std::vector<int> isolated_vertices {4, 2};
    std::vector<int> isolated_partition = PartitionPostIsolate(vertex_vertex, partition,
                                                               isolated_vertices);

    std::cout << "\nIsolated Partition: " << isolated_partition;

    for (auto&& isolated_vertex : isolated_vertices)
    {
        int isolated_part = isolated_partition[isolated_vertex];

        for (int i = 0; i < size; ++i)
        {
            if (isolated_partition[i] == isolated_part && i != isolated_vertex)
            {
                failed = true;
            }
        }

    }
    if (failed)
    {
        std::cout << "Metis Graph Partitioning Failed!\n";
    }

    // Test Weighted partition
    // Metis should avoid the natural option of cutting the middle connecting edge
    // since it is now heavily weighted
    {
        std::vector<double> weights(vertex_edge.Cols(), 1.0);
        weights[weights.size() / 2.0] = 10000;

        SparseMatrix M(std::move(weights));
        SparseMatrix A = RescaleLog(Mult(vertex_edge, M, edge_vertex));

        A.PrintDense("Weighted Vertex Vertex");

        bool contig = true;
        bool use_weight = true;

        std::vector<int> weighted_partition = Partition(A, 3, 2.0, contig, use_weight);
        std::cout << "\nWeighted Partition: " << weighted_partition;

        // Make sure vertices connected by mid edge are in same part
        int mid = size / 2;
        failed |= (weighted_partition[mid - 1] != weighted_partition[mid]);
    }

    return failed;
}
