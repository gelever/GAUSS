
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

/**
   @brief This test checks if the rescaling through MBuilder results in the same
          matrix as if the matrix is computed from scratch from the rescaled
          coefficient. This test requires the notion of finite volume and MFEM.
*/

#include "spe10.hpp"

using namespace rs2000;

smoothg::Vector SimpleAscendingScaling(int size)
{
    smoothg::Vector scale(size);

    for (int i = 0; i < size; i++)
    {
        scale[i] = i + 1;
    }

    return scale;
}

mfem::PWConstCoefficient InvElemScaleCoefficient(const smoothg::VectorView& elem_scale)
{
    mfem::Vector inverse_elem_scale(elem_scale.size());

    for (int elem = 0; elem < elem_scale.size(); ++elem)
    {
        inverse_elem_scale[elem] = 1.0 / elem_scale[elem];
    }
    return mfem::PWConstCoefficient(inverse_elem_scale);
}

smoothg::MixedMatrix OriginalScaledFineMatrix(mfem::ParFiniteElementSpace& sigmafespace,
                                              const smoothg::SparseMatrix& vertex_edge,
                                              const smoothg::VectorView& elem_scale)
{
    auto inv_scale_coef = InvElemScaleCoefficient(elem_scale);
    mfem::BilinearForm a2(&sigmafespace);
    a2.AddDomainIntegrator(new FiniteVolumeMassIntegrator(inv_scale_coef));

    int num_elem = sigmafespace.GetMesh()->GetNE();

    std::vector<smoothg::DenseMatrix> M_elem(num_elem);
    mfem::DenseMatrix M_el_i;

    for (int i = 0; i < num_elem; ++i)
    {
        a2.ComputeElementMatrix(i, M_el_i);

        int size = M_el_i.Height();
        M_elem[i].SetSize(size);

        for (int j = 0; j < size; ++j)
        {
            M_elem[i](j, j) = M_el_i(j, j);
        }
    }

    smoothg::ParMatrix edge_trueedge = ParMatrixToParMatrix(*sigmafespace.Dof_TrueDof_Matrix());
    smoothg::SparseMatrix D = smoothg::MixedMatrix::MakeLocalD(edge_trueedge, vertex_edge);
    smoothg::SparseMatrix W;

    return smoothg::MixedMatrix(std::move(M_elem), vertex_edge,
                                std::move(D), std::move(W),
                                std::move(edge_trueedge));
}

smoothg::SparseMatrix RescaledFineM(mfem::FiniteElementSpace& sigmafespace,
                                    const smoothg::VectorView& original_elem_scale,
                                    const smoothg::VectorView& additional_elem_scale)
{
    smoothg::Vector new_elem_scale(original_elem_scale);

    for (int i = 0; i < new_elem_scale.size(); ++i)
    {
        new_elem_scale[i] *= additional_elem_scale[i];
    }

    auto new_inv_scale_coef = InvElemScaleCoefficient(new_elem_scale);

    mfem::BilinearForm a1(&sigmafespace);
    a1.AddDomainIntegrator(new FiniteVolumeMassIntegrator(new_inv_scale_coef));
    a1.Assemble();
    a1.Finalize();

    return SparseToSparse(a1.SpMat());
}

smoothg::GraphCoarsen BuildCoarsener(const smoothg::Graph& graph,
                                     const smoothg::MixedMatrix& mixed_matrix)
{
    smoothg::GraphTopology gt(graph);
    smoothg::GraphSpace graph_space = smoothg::FineGraphSpace(graph);
    smoothg::Vector constant_rep(graph.vertex_edge_local_.Rows(),
                                 1.0 / std::sqrt(graph.global_vertices_));
    smoothg::SpectralPair spect_info{1.0, 3};

    return smoothg::GraphCoarsen(std::move(gt), graph_space, mixed_matrix, constant_rep, spect_info);
}

double FrobeniusNorm(MPI_Comm comm, const smoothg::SparseMatrix& mat)
{
    double frob_norm_square_loc = 0.0;

    const auto& data = mat.GetData();

    for (auto&& a_ij : data)
    {
        frob_norm_square_loc += a_ij * a_ij;
    }

    double frob_norm_square_global;
    MPI_Allreduce(&frob_norm_square_loc, &frob_norm_square_global, 1, MPI_DOUBLE, MPI_SUM, comm);

    return std::sqrt(frob_norm_square_global);
}

double RelativeDiff(MPI_Comm comm, const smoothg::SparseMatrix& M1, const smoothg::SparseMatrix& M2)
{
    smoothg::SparseMatrix diff = smoothg::Add(1.0, M1, -1.0, M2);
    return FrobeniusNorm(comm, diff) / FrobeniusNorm(comm, M1);
}

int main(int argc, char* argv[])
{
    // 1. Initialize MPI
    smoothg::MpiSession mpi_info(argc, argv);
    MPI_Comm comm = mpi_info.comm_;
    int myid = mpi_info.myid_;
    int num_procs = mpi_info.num_procs_;

    // Create a mesh graph, an edge fespace and a partition of the graph
    mfem::Mesh mesh(4, 4, 4, mfem::Element::HEXAHEDRON, 1);
    mfem::ParMesh pmesh(comm, mesh);

    for (int i = 0; i < pmesh.GetNE(); i++)
    {
        pmesh.SetAttribute(i, i + 1);
    }

    mfem::RT_FECollection sigmafec(0, pmesh.SpaceDimension());
    mfem::ParFiniteElementSpace sigmafespace(&pmesh, &sigmafec);

    smoothg::SparseMatrix vertex_edge = TableToSparse(pmesh.ElementToFaceTable());
    smoothg::SparseMatrix edge_bdratt = GenerateBoundaryAttributeTable(pmesh);
    smoothg::ParMatrix edge_trueedge = ParMatrixToParMatrix(*sigmafespace.Dof_TrueDof_Matrix());

    int coarsening_factor = 8;
    std::vector<int> partitioning = smoothg::PartitionAAT(vertex_edge, coarsening_factor);

    int num_parts = *std::max_element(std::begin(partitioning), std::end(partitioning)) + 1;

    //Create simple element and aggregate scaling
    smoothg::Vector elem_scale = SimpleAscendingScaling(pmesh.GetNE());
    smoothg::Vector agg_scale = SimpleAscendingScaling(num_parts);

    // Create a fine level MixedMatrix corresponding to piecewise constant coefficient
    smoothg::Graph graph(vertex_edge, edge_trueedge, partitioning);
    smoothg::MixedMatrix fine_mgL = OriginalScaledFineMatrix(sigmafespace, vertex_edge, elem_scale);
    fine_mgL.AssembleM();

    // Create a coarsener to build interpolation matrices and coarse M builder
    smoothg::GraphCoarsen coarsener = BuildCoarsener(graph, fine_mgL);
    auto coarse_mgL = coarsener.Coarsen(fine_mgL);

    // Interpolate agg scaling (coarse level) to elements (fine level)
    smoothg::SparseMatrix P = smoothg::MakeAggVertex(partitioning);
    smoothg::Vector interp_agg_scale = P.MultAT(agg_scale);

    // Assemble rescaled fine and coarse M through MixedMatrix
    fine_mgL.AssembleM(interp_agg_scale.data());
    auto fine_M1 = fine_mgL.LocalM();

    coarse_mgL.AssembleM(agg_scale.data());
    auto coarse_M1 = coarse_mgL.LocalM();

    // Assembled rescaled fine and coarse M through direct assembling and RAP
    auto fine_M2 = RescaledFineM(sigmafespace, elem_scale, interp_agg_scale);

    const auto& Psigma = coarsener.Pedge();
    const auto& Psigma_T = Psigma.Transpose();
    smoothg::SparseMatrix coarse_M2 = smoothg::Mult(Psigma_T, fine_M2, Psigma);

    // Check relative differences measured in Frobenius norm
    double fine_diff = RelativeDiff(comm, fine_M1, fine_M2);
    double coarse_diff = RelativeDiff(comm, coarse_M1, coarse_M2);

    double tol = 1e-14;
    bool failed = false;

    if (fine_diff > tol)
    {
        ParPrint(myid, std::cerr << "Fine level rescaling is NOT working as expected: "
                 << fine_diff << "\n");
        failed = true;
    }

    if (coarse_diff > tol)
    {
        ParPrint(myid, std::cerr << "Coarse level rescaling is NOT working as expected: "
                 << coarse_diff << "\n");
        failed = true;
    }

    return failed;
}

