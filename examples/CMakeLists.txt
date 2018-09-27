#!/bin/sh
# BHEADER ####################################################################
#
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
#
# This file is part of smoothG. For more information and source code
# availability, see https://www.github.com/llnl/smoothG.
#
# smoothG is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.
#
#################################################################### EHEADER #

add_executable(generalgraph generalgraph.cpp)
target_link_libraries(generalgraph GAUSS)

add_executable(graph_timestep graph_timestep.cpp)
target_link_libraries(graph_timestep GAUSS)

add_executable(graphupscale graphupscale.cpp)
target_link_libraries(graphupscale GAUSS)

add_executable(poweriter poweriter.cpp)
target_link_libraries(poweriter GAUSS)

add_executable(graph_mltopo graph_mltopo.cpp)
target_link_libraries(graph_mltopo GAUSS)

add_executable(graph_mlmc graph_mlmc.cpp)
target_link_libraries(graph_mlmc GAUSS)

add_executable(mleigen mleigen.cpp)
target_link_libraries(mleigen GAUSS)

add_executable(sampler sampler.cpp)
target_link_libraries(sampler GAUSS)

if (NOT DEFINED GAUSS_TEST_TOL)
    set(GAUSS_TEST_TOL 1e-4)
endif()
if (NOT DEFINED GAUSS_TEST_PROCS)
    set(GAUSS_TEST_PROCS 4)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/examples/stest.py"
  "${PROJECT_BINARY_DIR}/examples/stest.py" @ONLY)

add_test(eigenvector1 python stest.py eigenvector1)
add_test(eigenvector4 python stest.py eigenvector4)
add_test(fv-hybridization python stest.py fv-hybridization)
add_test(slice19 python stest.py slice19)

add_test(samplegraph1 python stest.py samplegraph1)
add_test(samplegraph4 python stest.py samplegraph4)
add_test(graph-hybridization python stest.py graph-hybridization)
add_test(graph-weight python stest.py graph-weight)
add_test(timestep python stest.py timestep)
add_test(graphupscale graphupscale)
add_test(graph_mltopo graph_mltopo)
add_test(graph_mlmc graph_mlmc)
add_test(mleigen mleigen)

add_test(pareigenvector1 python stest.py pareigenvector1)
add_test(pareigenvector4 python stest.py pareigenvector4)
add_test(parfv-hybridization python stest.py parfv-hybridization)
add_test(parslice19 python stest.py parslice19)

add_test(parsamplegraph1 python stest.py parsamplegraph1)
add_test(parsamplegraph4 python stest.py parsamplegraph4)
add_test(pargraph-hybridization python stest.py pargraph-hybridization)
add_test(partimestep python stest.py partimestep)
add_test(pargraphupscale mpirun -np ${GAUSS_TEST_PROCS} ./graphupscale)
add_test(parmltopo mpirun -np ${GAUSS_TEST_PROCS} ./graph_mltopo)
add_test(parmlmc mpirun -np ${GAUSS_TEST_PROCS} ./graph_mlmc)
add_test(parmleigen mpirun -np ${GAUSS_TEST_PROCS} ./mleigen)

if (APPLE)
  add_test(graph-metis python stest.py graph-metis-mac)

  # TODO(gelever1): figure out why these are so different on osx
  #add_test(graph-usegenerator python stest.py graph-usegenerator-mac)
  #add_test(pargraph-usegenerator python stest.py pargraph-usegenerator-mac)

  add_test(pargraph-metis python stest.py pargraph-metis-mac)
else()

  # NOTE(gelever1): ML tests depend on metis to coarsen topology
  # but metis produces different results on APPLE
  # so these tests only hold on linux
  #add_test(ml_eigenvector1 python stest.py ml_eigenvector1)
  #add_test(ml_eigenvector4 python stest.py ml_eigenvector4)
  #add_test(ml_fv-hybridization python stest.py ml_fv-hybridization)
  #add_test(ml_slice19 python stest.py ml_slice19)
  #add_test(ml_fv-metis python stest.py ml_fv-metis)

  #add_test(ml_pareigenvector1 python stest.py ml_pareigenvector1)
  #add_test(ml_pareigenvector4 python stest.py ml_pareigenvector4)
  #add_test(ml_parfv-hybridization python stest.py ml_parfv-hybridization)
  #add_test(ml_parslice19 python stest.py ml_parslice19)

  add_test(fv-metis python stest.py fv-metis)
  add_test(graph-metis python stest.py graph-metis)
  add_test(graph-usegenerator python stest.py graph-usegenerator)
  add_test(pargraph-usegenerator python stest.py pargraph-usegenerator)
  add_test(pargraph-metis python stest.py pargraph-metis)

  add_test(veigenvector python stest.py veigenvector)
  add_test(vgraph-small-usegenerator python stest.py vgraph-small-usegenerator)
  add_test(vgraph-small-usegenerator-hb python stest.py vgraph-small-usegenerator-hb)

  # TODO(gelever1): figure out why these are so different on osx
  add_test(poweriter python stest.py poweriter)
  add_test(parpoweriter python stest.py parpoweriter)
endif()

#add_test(isolate_coarsen python stest.py isolate-coarsen)
