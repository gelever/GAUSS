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

option(SMOOTHG_ENABLE_MFEM_MINIAPPS "Should miniapps that depend on MFEM be enabled?" NO)
option(SMOOTHG_ENABLE_PARELAGMC_MINIAPPS "Should miniapps that depend on ParELAGMC be enabled?" NO)

add_subdirectory(graphvis)

if (SMOOTHG_ENABLE_MFEM_MINIAPPS)
    add_subdirectory(rs2001)

    if (SMOOTHG_ENABLE_PARELAGMC_MINIAPPS)
        add_subdirectory(MLMC)
    endif()
endif()
