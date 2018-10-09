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

# add executables
add_executable(tinygraphsolver tinygraphsolver.cpp)
target_link_libraries(tinygraphsolver GAUSS)

add_executable(eigen eigen.cpp)
target_link_libraries(eigen GAUSS)

add_executable(test_MetisGraphPartitioner test_MetisGraphPartitioner.cpp)
target_link_libraries(test_MetisGraphPartitioner GAUSS)

add_executable(wattsstrogatz wattsstrogatz.cpp)
target_link_libraries(wattsstrogatz GAUSS)

add_executable(test_Graph test_Graph.cpp)
target_link_libraries(test_Graph GAUSS)

add_executable(test_projection test_projection.cpp)
target_link_libraries(test_projection GAUSS)

#add_executable(test_Solvers test_Solvers.cpp)
#target_link_libraries(test_Solvers GAUSS)

# add tests
# add_test(lineargraph lineargraph)
# add_test(lineargraph64 lineargraph --size 64)
# add_valgrind_test(vlineargraph lineargraph)


add_test(tinygraphsolver tinygraphsolver)
add_test(tinygraphsolver_wg tinygraphsolver -wg)
add_test(tinygraphsolver_wb tinygraphsolver -wb)
add_test(tinygraphsolver_wg_wb tinygraphsolver -wg -wb)

add_test(partinygraphsolver mpirun -np 2 ./tinygraphsolver)
add_test(partinygraphsolver_wg mpirun -np 2 ./tinygraphsolver -wg)
add_test(partinygraphsolver_wb mpirun -np 2 ./tinygraphsolver -wb)
add_test(partinygraphsolver_wg_wb mpirun -np 2 ./tinygraphsolver -wg -wb)

add_test(eigen eigen)

add_test(test_MetisGraphPartitioner test_MetisGraphPartitioner)

add_test(test_Graph test_Graph)
add_test(parttest_Graph mpirun -np 2 ./test_Graph)

add_test(test_projection test_projection)
add_test(parttest_projection mpirun -np 2 ./test_projection)

# add_test(test_IsolatePartitioner test_IsolatePartitioner)
# add_valgrind_test(vtest_IsolatePartitioner test_IsolatePartitioner)

add_test(wattsstrogatz wattsstrogatz)
add_test(parwattsstrogatz mpirun -np 2 ./wattsstrogatz)

if (NOT APPLE)
    add_valgrind_test(vtinygraph tinygraphsolver)
    add_valgrind_test(vtinygraph_wg tinygraphsolver -wg)
    add_valgrind_test(vtinygraph_wb tinygraphsolver -wb)
    add_valgrind_test(vtinygraph_wg_wb tinygraphsolver -wg -wb)
    add_valgrind_test(veigen eigen)
    add_valgrind_test(vtest_MetisGraphPartitioner test_MetisGraphPartitioner)
    add_valgrind_test(vwattsstrogatz wattsstrogatz)
endif()

add_test(NAME style
    COMMAND ${ASTYLE_COMMAND} --options=GAUSS.astylerc --dry-run src/*.?pp include/*.?pp examples/*.?pp testcode/*.?pp
    WORKING_DIRECTORY ${GAUSS_SOURCE_DIR})
set_tests_properties(style
  PROPERTIES FAIL_REGULAR_EXPRESSION
  "Formatted")
