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

###############################################################
# MIT License
# 
# Copyright (c) 2018 Pablo Arias
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
# OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# 
###############################################################

###############################################################
# Using https://github.com/pabloariasal/modern-cmake-sample
# Modified by gelever on April 1st, 2018 #
###############################################################

cmake_minimum_required(VERSION 3.5)
project(GAUSS VERSION 1.0 LANGUAGES CXX)
enable_testing()

option(GAUSS_USE_ARPACK "Should ARPACK be enabled?" NO)

if (GAUSS_USE_ARPACK)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
    find_package(ARPACK REQUIRED)
    LIST(REMOVE_AT CMAKE_MODULE_PATH -1)
endif()

find_package(linalgcpp REQUIRED)

add_library(GAUSS
    src/GraphCoarsen.cpp
    src/Graph.cpp
    src/GraphEdgeSolver.cpp
    src/GraphGenerator.cpp
    src/GraphTopology.cpp
    src/GraphUpscale.cpp
    src/HybridSolver.cpp
    src/LocalEigenSolver.cpp
    src/MGLSolver.cpp
    src/MinresBlockSolver.cpp
    src/MixedMatrix.cpp
    src/SharedEntityComm.cpp
    src/SPDSolver.cpp
    src/Utilities.cpp
)

target_compile_options(GAUSS PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -O2>)
target_compile_options(GAUSS PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -O2>)

target_link_libraries(GAUSS
    PUBLIC
        linalgcpp::partition
        linalgcpp::sparsesolve
        linalgcpp::parlinalgcpp
        linalgcpp::linalgcpp
        $<$<BOOL:${GAUSS_USE_ARPACK}>:Arpack::Arpack>
)

target_include_directories(GAUSS
    PUBLIC
		$<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)

# Astyle
find_program(ASTYLE_COMMAND astyle HINTS ${ASTYLE_DIR})
add_custom_target(style
    ${ASTYLE_COMMAND} --options=GAUSS.astylerc src/*.cpp include/*.hpp examples/*.?pp testcode/*.?pp
    WORKING_DIRECTORY ${GAUSS_SOURCE_DIR}
    COMMENT "Formating source code" VERBATIM
    )

# Valgrind
find_program(MEMORYCHECK_COMMAND valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "")
list(APPEND MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full")
list(APPEND MEMORYCHECK_COMMAND_OPTIONS "--error-exitcode=1")
function(add_valgrind_test name binary)
  add_test(${name} ${MEMORYCHECK_COMMAND} ${MEMORYCHECK_COMMAND_OPTIONS} ./${binary})
endfunction(add_valgrind_test)

# Installation
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libGAUSS)

install(TARGETS GAUSS
    EXPORT GAUSS-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets to a script
install(EXPORT GAUSS-targets
    FILE GAUSSTargets.cmake
    NAMESPACE GAUSS::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/GAUSSConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/GAUSSConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/GAUSSConfig.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

configure_file(
    "${PROJECT_SOURCE_DIR}/config/GAUSS_config.h.in"
    "${PROJECT_BINARY_DIR}/GAUSS_config.h"
  )

# Install Config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/GAUSSConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/GAUSSConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Export from the build tree
export(EXPORT GAUSS-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/GAUSSTargets.cmake
    NAMESPACE GAUSS::
       )

   export(PACKAGE GAUSS)

# Sub Directories
add_subdirectory(testcode)
add_subdirectory(examples)
add_subdirectory(miniapps)
