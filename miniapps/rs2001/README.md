GAUSS/RS2001
=================

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

RS2001 demonstrates multilevel upscaling using MFEM and the SPE10 dataset.
Several examples are provided:

| Name        | Description |
| ----------- |-------------|
| `finitevolume.cpp` | Finite volume upscaling |
| `dirichlet.cpp` | Application of Dirichlet boundary conditions |
| `MLMC_SPE10.cpp`| Sample generation and element scaling |
| `PDESamplerTest.cpp` | Generate samples and statistics|
| `timestep.cpp` | Diffusion over time |
