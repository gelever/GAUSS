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
project(ReservoirSimulator2000 VERSION 1.0 LANGUAGES CXX)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
find_package(MFEM REQUIRED)

add_library(ReservoirSimulator2000
    src/spe10.cpp
    src/Sampler.cpp
    src/DarcySolver.cpp
    src/Utilities.cpp
)

add_dependencies(ReservoirSimulator2000 smoothG)

set_target_properties(ReservoirSimulator2000 PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

target_compile_options(ReservoirSimulator2000 PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic -O2>)
target_compile_options(ReservoirSimulator2000 PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic -O2>)

target_link_libraries(ReservoirSimulator2000
    PUBLIC
        mfem::mfem
        smoothG
)

target_include_directories(ReservoirSimulator2000
    PUBLIC
		$<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
)

# Astyle
find_program(ASTYLE_COMMAND astyle HINTS ${ASTYLE_DIR})
add_custom_target(rs2001_style
    ${ASTYLE_COMMAND} --options=rs2001.astylerc src/*.cpp include/*.hpp examples/*.?pp testcode/*.?pp
    WORKING_DIRECTORY ${ReservoirSimulator2000_SOURCE_DIR}
    COMMENT "Formating source code" VERBATIM
    )

# Installation
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/libsmoothg)

install(TARGETS ReservoirSimulator2000
    EXPORT ReservoirSimulator2000-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets to a script
install(EXPORT ReservoirSimulator2000-targets
    FILE ReservoirSimulator2000Targets.cmake
    NAMESPACE ReservoirSimulator2000::
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Create Config files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/ReservoirSimulator2000ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/ReservoirSimulator2000Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ReservoirSimulator2000Config.cmake
    INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

configure_file(
    "${PROJECT_SOURCE_DIR}/config/ReservoirSimulator2000_config.h.in"
    "${PROJECT_BINARY_DIR}/ReservoirSimulator2000_config.h"
  )

# Install Config files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ReservoirSimulator2000Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/ReservoirSimulator2000ConfigVersion.cmake
    DESTINATION ${INSTALL_CONFIGDIR}
)

# Export from the build tree
export(EXPORT ReservoirSimulator2000-targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/ReservoirSimulator2000Targets.cmake
       NAMESPACE ReservoirSimulator2000::
       )

   export(PACKAGE ReservoirSimulator2000)

# Sub Directories
add_subdirectory(examples)
add_subdirectory(testcode)
