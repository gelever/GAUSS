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

Installing smoothG            {#INSTALL}
==========

The following instructions will install smoothG and all of its
dependencies.

# Dependencies:

* [linalgcpp](https://github.com/gelever/linalgcpp)  - Serial linear algebra and solvers
   * [blas](http://www.netlib.org/blas/) - Dense matrix operations
   * [lapack](http://www.netlib.org/lapack/) - Dense matrix solvers
   * [hypre](https://github.com/LLNL/hypre) - Distrubuted linear algebra and solvers
   * [SuiteSparse/UMFPACK](http://faculty.cse.tamu.edu/davis/suitesparse.html)
   * [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) - Graph partitioner
* [ARPACK](https://www.caam.rice.edu/software/ARPACK/) - Sparse EigenSolver (optional)

# Build Dependencies:

These instructions will build dependencies in the your home folder: `${HOME}`

If not installing in standard system locations (`/usr/`, `/usr/local/`, etc),
you will need to export the appropriate `LIBRARY_PATH` and `LD_LIBRARY_PATH`
so that the linker/loader can find them.

For example the final `LIBRARY_PATH` will look like:

    export LD_LIBRARY_PATH=${HOME}/local/lib:$LD_LIBRARY_PATH


## blas

    check if exists or install from package manager

## lapack

    check if exists or install from package manager

## metis-5.1.0

    tar -xvzf metis-5.1.0.tar.gz
    cd metis-5.1.0

    make config prefix=${HOME}/metis
    make install

## hypre-2.10.0b

    tar -xvfz hypre-2.10.0b.gz
    cd hypre-2.10.0b/src

    ./configure --disable-fortran --prefix=${HOME}/hypre
    make install

## SuiteSparse-4.5.4

    tar -xvfz SuiteSparse-4.5.4.tar.gz
    cd SuiteSparse-4.5.4

    make install BLAS=/usr/lib64/libblas.so.3 LAPACK=/usr/lib64/liblapack.so.3 \
        INSTALL=${HOME}/SuiteSparse

    #(Replace blas and lapack library locations appropriately)

## linalgcpp
    
    git clone -b develop https://github.com/gelever/linalgcpp.git linalgcpp
    cd linalgcpp
    mkdir -p build && cd build
    CXX=mpic++ CC=mpicc cmake .. \
        -DLINALGCPP_ENABLE_MPI=Yes \
        -DLINALGCPP_ENABLE_METIS=Yes \
        -DLINALGCPP_ENABLE_SUITESPARSE=Yes \
        -DHypre_DIR=${HOME}/hypre \
        -DMETIS_DIR=${HOME}/metis \
        -DSUITESPARSE_INCLUDE_DIR_HINTS=${HOME}/SuiteSparse/include \
        -DSUITESPARSE_LIBRARY_DIR_HINTS=${HOME}/SuiteSparse/lib \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/linalgcpp
    make -j3 install

# Optional Dependencies:

* [Valgrind](http://valgrind.org/)
* [ARPACK](https://www.caam.rice.edu/software/ARPACK/)

## Valgrind

    tar -xvf valgrind-3.12.0
    cd valgrind-3.12.0

    ./configure --prefix=${HOME}/valgrind
    make
    make install

# Build smoothG

Clone the smoothG repo and cd into smoothG directory.

To build smoothG, either copy, modify, and run a config file from config/
or pass the parameters directly to cmake:

    mkdir -p build
    cd build

    CC=mpicc CXX=mpic++ cmake \
        -DMETIS_DIR=${HOME}/metis \
        -DHypre_INC_DIR=${HOME}/hypre/include \
        -DHypre_LIB_DIR=${HOME}/hypre/lib \
        -DSUITESPARSE_INCLUDE_DIR_HINTS=${HOME}/SuiteSparse/include \
        -DSUITESPARSE_LIBRARY_DIR_HINTS=${HOME}/SuiteSparse/lib \
        ${BASE_DIR} \
        ${EXTRA_ARGS}

    make
    make test
    make doc

# Notes:

Metis gives you the option of choosing between float and double
as your real type by altering the REALTYPEWIDTH constant in
metis.h. To pass our tests, you need to have REALTYPEWIDTH set to 32
(resulting in float for your real type).

