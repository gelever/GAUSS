#!/bin/sh

# Downloads and builds MFEM
if [ -z "$INSTALL_DIR" ]; then INSTALL_DIR=${PWD}/extern; fi
if [ -z "$METIS_DIR" ]; then METIS_DIR=$INSTALL_DIR/metis; fi
if [ -z "$SUITESPARSE_DIR" ]; then SUITESPARSE_DIR=$INSTALL_DIR/SuiteSparse; fi
if [ -z "$HYPRE_DIR" ]; then HYPRE_DIR=$INSTALL_DIR/hypre; fi

TMP_DIR=/tmp/mfem

mkdir -p $TMP_DIR
cd $TMP_DIR

git clone https://github.com/mfem/mfem.git mfem
cd mfem
git checkout v3.3.2
make config \
    MFEM_USE_METIS_5=YES \
    MFEM_USE_LAPACK=YES \
    MFEM_USE_SUITESPARSE=YES \
    MFEM_USE_MPI=YES \
    HYPRE_DIR=${HYPRE_DIR} \
    SUITESPARSE_DIR=${SUITESPARSE_DIR} \
    METIS_DIR=${METIS_DIR} \
    PREFIX=${INSTALL_DIR}/mfem

CC=mpicc CXX=mpic++ make -j3 install PREFIX=${INSTALL_DIR}/mfem

rm -r $TMP_DIR
