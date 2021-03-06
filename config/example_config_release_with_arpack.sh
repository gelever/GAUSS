#!/bin/sh
# BHEADER ####################################################################
#
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-759464. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of GAUSS. For more information and source code
# availability, see https://www.github.com/gelever/GAUSS.
#
# GAUSS is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
#################################################################### EHEADER #


# This is the path to the root of the git repo
# the BASE_DIR should contain config/GAUSS_config.h.in
BASE_DIR=/path/to/the/root/directory/of/GAUSS

# this is where we actually build binaries and so forth
BUILD_DIR=${BASE_DIR}/build

EXTRA_ARGS=$@

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Force a reconfigure
rm CMakeCache.txt
rm -rf CMakeFiles

cmake \
    -DMFEM_DIR=/path/to/the/directory/where/mfem/is/installed \
    -DMETIS_DIR=/path/to/the/directory/where/metis/is/installed \
    -DHYPRE_DIR=/path/to/the/directory/where/hypre/is/installed \
    -DSuiteSparse_DIR=/path/to/the/directory/where/SuiteSparse/is/installed \
    -DSPE10_DIR=/path/to/the/directory/where/spe_perm.dat/is/located \
    \
    -DUSE_ARPACK=ON \
    -DARPACK_DIR=/path/to/the/lib/directory/of/arpack \
    -DARPACKPP_DIR=/path/to/the/root/directory/of/arpackpp \
    \
    -DCMAKE_BUILD_TYPE=Release \
    -DBLAS_LIBRARIES=/path/to/the/blas/library/file \
    -DLAPACK_LIBRARIES=/path/to/the/lapack/library/file \
    \
    ${EXTRA_ARGS} \
    ${BASE_DIR}
