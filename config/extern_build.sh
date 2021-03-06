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
BASE_DIR=${PWD}

# this is where we actually build binaries and so forth
BUILD_DIR=${BASE_DIR}/build

EXTRA_ARGS=$@

# Force a reconfigure
rm CMakeCache.txt
rm -rf CMakeFiles

# Build Options
EXTERN_DIR=${PWD}/extern
METIS_DIR=$EXTERN_DIR/metis
HYPRE_DIR=$EXTERN_DIR/hypre
SUITESPARSE_DIR=$EXTERN_DIR/SuiteSparse
ASTYLE_DIR=$EXTERN_DIR/astyle
USE_ARPACK=No
BUILD_TYPE=Debug
TEST_TOL=1e-4
TEST_PROCS=2

mkdir -p $BUILD_DIR
cd $BUILD_DIR

CC=mpicc CXX=mpic++ cmake \
    -DMETIS_DIR=$METIS_DIR \
    -DHypre_INC_DIR=$HYPRE_DIR/include \
    -DHypre_LIB_DIR=$HYPRE_DIR/lib \
    -DSUITESPARSE_INCLUDE_DIR_HINTS=$SUITESPARSE_DIR/include \
    -DSUITESPARSE_LIBRARY_DIR_HINTS=$SUITESPARSE_DIR/lib \
    -DASTYLE_DIR=$ASTYLE_DIR/bin \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DGAUSS_USE_ARPACK=$USE_ARPACK \
    -DGAUSS_TEST_TOL=$TEST_TOL \
    -DGAUSS_TEST_PROCS=$TEST_PROCS \
    $BASE_DIR \
    $EXTRA_ARGS

make -j 3
