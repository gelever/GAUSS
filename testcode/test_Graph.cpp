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
   Test Graph object, distributed vs global

   edge maps should be the same

   vertex maps don't have to be the same,
   but should assign each vertex a unique global id
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
    MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    assert(num_procs == 1 || num_procs == 2);

    // load the graph
    std::string graph_filename = "../../graphdata/vertex_edge_tiny.txt";
    SparseMatrix vertex_edge = ReadCSR(graph_filename);

    std::vector<int> partition {0, 0, 0, 1, 1, 1};

    Graph global_graph(comm, vertex_edge, partition);
    Graph dist_graph(global_graph.vertex_edge_local_,
                     global_graph.edge_true_edge_,
                     global_graph.part_local_);

    bool failed = false;

    // TODO(gelever1): Add some test here

    if (myid == 0)
    {
        std::cout << "Global Vertex Map: " << global_graph.vertex_map_;
        std::cout << "Dist.  Vertex Map: " << dist_graph.vertex_map_;
    }

    return failed;
}

