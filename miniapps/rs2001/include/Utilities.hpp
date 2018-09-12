/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-745247. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of smoothG. For more information and source code
 * availability, see https://www.github.com/llnl/smoothG.
 *
 * smoothG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

/**
   @file Utilities.hpp
   @brief Helper functions
*/

#include "mfem.hpp"
#include "smoothG.hpp"

using std::unique_ptr;

namespace rs2000
{

std::vector<int> MetisPart(const smoothg::SparseMatrix& vertex_edge, int num_parts);

void VectorToField(const smoothg::VectorView& vect, mfem::GridFunction& field);
smoothg::Vector VectorToVector(const mfem::Vector& mfem_vector);
mfem::Vector VectorToVector(const smoothg::VectorView& vector);
smoothg::SparseMatrix SparseToSparse(const mfem::SparseMatrix& sparse);
smoothg::SparseMatrix TableToSparse(const mfem::Table& table);
smoothg::ParMatrix ParMatrixToParMatrix(const mfem::HypreParMatrix& mat);

smoothg::SparseMatrix GenerateBoundaryAttributeTable(const mfem::Mesh& mesh);

using VisRange = std::pair<double, double>;

VisRange GetVisRange(MPI_Comm comm, const mfem::Vector& vect);
VisRange GetVisRange(MPI_Comm comm, const smoothg::VectorView& vect);
enum class VisType { Angle, Top, Big};


// Setup and reuse the same visualization window
void VisSetup(MPI_Comm comm, mfem::socketstream& vis_v,
              mfem::ParGridFunction& field,
              mfem::ParMesh& pmesh, VisRange range,
              const std::string& title = "",
              const std::string& caption = "",
              bool show_log = false,
              VisType vis_type = VisType::Top);

void VisUpdate(MPI_Comm comm, mfem::socketstream& vis_v,
               mfem::ParGridFunction& field,
               mfem::ParMesh& pmesh);

// Create seperate visualization window
void Visualize(const mfem::Vector& sol, mfem::ParMesh& pmesh, mfem::ParGridFunction& field,
               VisRange vis_range, const std::string& title, int level, bool show_log = false,
               VisType vis_type = VisType::Top);


std::vector<int> MetisPart(mfem::ParFiniteElementSpace& sigmafespace,
                           mfem::ParFiniteElementSpace& ufespace,
                           mfem::Array<double>& coarsening_factor);

std::vector<int> CartPart(std::vector<int>& num_procs_xyz,
                          mfem::ParMesh& pmesh, mfem::Array<double>& coarsening_factor);

void EliminateEssentialBC(smoothg::GraphUpscale& upscale,
                          const smoothg::SparseMatrix& bdr_attr_vertex,
                          const std::vector<int>& ess_bdr,
                          const smoothg::BlockVector& x,
                          smoothg::BlockVector& b);


/// Visualization short cuts:
constexpr auto vis_flatten = "keys ------\n";
constexpr auto vis_top_view = "view 0 0\n";
constexpr auto vis_angle_view = "view 45 -45\n";
constexpr auto vis_embiggen = "keys ]]]]]\n";
constexpr auto vis_mesh_blend = "keys ff\n";
constexpr auto vis_mesh_level_lines = "keys mm\n";
constexpr auto vis_cet_rainbow_colormap = "keys PPPPP\n";
constexpr auto vis_log_display = "keys L\n";
constexpr auto vis_colorbar = "keys c\n";
constexpr auto vis_pause = "pause\n";
constexpr auto vis_screenshot = "keys S\n";
constexpr auto vis_light = "keys l\n";
constexpr auto vis_perspective = "keys j\n";

} // namespace rs2000

#endif /* UTILITES_HPP_ */


