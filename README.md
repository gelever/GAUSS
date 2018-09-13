smoothG [![Build Status](https://travis-ci.org/gelever/smoothG.svg?branch=master)](https://travis-ci.org/gelever/smoothG)
=================

<!-- BHEADER ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 +
 + Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 + Produced at the Lawrence Livermore National Laboratory.
 + LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 +
 + This file is part of smoothG. For more information and source code
 + availability, see https://www.github.com/llnl/smoothG.
 +
 + smoothG is free software; you can redistribute it and/or modify it under the
 + terms of the GNU Lesser General Public License (as published by the Free
 + Software Foundation) version 2.1 dated February 1999.
 +
 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ EHEADER -->

<!-- !["smoothG logo](doc/smoothg_logo.png) -->

Mixed graph-Laplacian upscaling and solvers.

# Overview

This project is intended to take a graph and build a smaller (upscaled)
graph that is representative of the original in some way. We represent
the graph Laplacian in a mixed form, solve some local eigenvalue problems
to uncover near-nullspace modes, and use those modes as coarse degrees
of freedom.

This project was originally a fork of [LLNL/smoothG](https://github.com/llnl/smoothG).
However, it has been rewritten completely to remove dependence on the finite element library [mfem](https://github.com/llnl/mfem).
Apart from now being completely algebraic, there are some additional extensions.
This includes multilevel upscaling, refactored code base, additional build scripts,
more robust continuous integration environment, and so on and so on.

This code is based largely on the following paper:

> A.T. Barker, C.S. Lee, and P.S. Vassilevski, Spectral upscaling for 
> graph Laplacian problems with application to reservoir simulation, *SIAM
> J. Sci. Comput.*, Vol. 39, No. 5, pp. S323–S346.

# Examples
There are some simple setup steps required to start obtaining upscaled solutions:
```c++
{
    // Input data
    SparseMatrix vertex_edge = ReadCSR(graph_filename);
    std::vector<int> partition = ReadText<int>(part_filename);
    
    // Perform coarsening setup
    Graph graph(comm, vertex_edge, partition);
    GraphUpscale upscale(graph, {spect_tol, max_evects});

    // Apply upscaling operation
    Vector rhs = ReadVertexVector(graph, rhs_filename);
    Vector sol = upscale.Solve(rhs);
}
```
The `Graph` object handles the fine level input graph information.  Both serial and distributed graph inputs are supported.
In this case, `vertex_edge` and `partition` have been read in from file.

The `GraphUpscale` object handles the coarsening procedure and the application of upscaling.
There are several user specified parameters that can control the coarsening process.

For more detailed walkthroughs, see [EXAMPLE.md](doc/EXAMPLE.md) or the examples directory [README.md](examples/README.md).

# Installation
There are several dependencies required:
### Linear Algebra:
* [linalgcpp](https://github.com/gelever/linalgcpp)  - Serial linear algebra and solvers
   * [blas](http://www.netlib.org/blas/) - Dense matrix operations
   * [lapack](http://www.netlib.org/lapack/) - Dense matrix solvers
* [parlinalgcpp](https://github.com/gelever/parlinalgcpp) - Wrapper for hypre
   * [hypre](https://github.com/LLNL/hypre) - Distrubuted linear algebra and solvers
* [sparsesolver](https://github.com/gelever/sparsesolver) - Wrapper for SuiteSparse
   * [SuiteSparse/UMFPACK](http://faculty.cse.tamu.edu/davis/suitesparse.html)
* [partition](https://github.com/gelever/partition) - Wrapper for METIS
   * [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) - Graph partitioner
* [ARPACK](https://www.caam.rice.edu/software/ARPACK/) - Sparse EigenSolver (optional)

These modules may be collapsed into one project in the future.
   
### Other:
* [CMake](https://cmake.org/)  - Build generator

For more detailed installation instructions, see [INSTALL.md](INSTALL.md).

# Project Structure
```sh
mysmoothg
├── config    - Configuration and build scripts
├── examples  - Graph upscaling examples
├── graphdata - Example graphs, partitions, and vectors
├── miniapps  - Applications that may have external dependencies (optional)
├── testcode  - Verification of library functions
├── src       - Library implementation
└── include   - Library declarations
```

# Contributions
This code has contributions from:
- Andrew T. Barker (atb@llnl.gov)
- Stephan Gelever (gelever1@llnl.gov)
- Chak Shing Lee (cslee@llnl.gov)
- Colin Ponce (ponce11@llnl.gov)


# Copyright
Copyright (c) 2018, Lawrence Livermore National Security, LLC.
This work was performed under the auspices of the U.S. Department of Energy by
Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.
LLNL-CODE-745247.
