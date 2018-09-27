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

# Apply flags to all test targets
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CXX_STANDARD_REQUIRED YES)
# set(CMAKE_CXX_EXTENSIONS NO)

# add executables
add_executable(rescaling rescaling.cpp)
target_link_libraries(rescaling ReservoirSimulator2001)

add_test(rescaling rescaling)
add_test(parrescaling mpirun -np 2 ./rescaling)

if (NOT APPLE)
    add_valgrind_test(vrescaling rescaling)
endif()

add_test(NAME rs2001_style
  COMMAND ${ASTYLE_COMMAND} --options=rs2001.astylerc --dry-run include/*.?pp src/*.?pp examples/*.?pp testcode/*.?pp
  WORKING_DIRECTORY ${ReservoirSimulator2001_SOURCE_DIR})
set_tests_properties(rs2001_style
  PROPERTIES FAIL_REGULAR_EXPRESSION
  "Formatted")
