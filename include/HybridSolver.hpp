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

/** @file

    @brief Routines for setup and implementing the hybridization solver.

           The setup involves forming the hybridized system and constructing a
           preconditioner for the hybridized system.

           In the solving phase (Mult), a given right hand side is transformed
           and the hybridized system is solved by calling an iterative method
           preconditioned by the preconditioner constructed in the setup.
           Lastly, the solution of the original system is computed from the
           solution (Lagrange multiplier) of the hybridized system through
           back substition.
*/

#ifndef __HYBRIDSOLVER_HPP
#define __HYBRIDSOLVER_HPP

#include "Utilities.hpp"
#include "MGLSolver.hpp"
#include "GraphCoarsen.hpp"
#include "MixedMatrix.hpp"

namespace smoothg
{

/**
   @brief Hybridization solver for saddle point problems

   This solver is intended to solve saddle point problems of the form
   \f[
     \left( \begin{array}{cc}
       M&  D^T \\
       D&  0
     \end{array} \right)
     \left( \begin{array}{cc}
       \sigma \\
       u
     \end{array} \right)
     =\left( \begin{array}{cc}
       0 \\
       f
     \end{array} \right)
   \f]

   Given \f$ \widehat{M}, \widehat{D} \f$, the "element" matrices of
   \f$M\f$ and \f$D\f$, the following hybridized system is formed

   \f[
     H = C (\widehat{M}^{-1}-\widehat{M}^{-1}\widehat{D}^T
           (\widehat{D}\widehat{M}^{-1}\widehat{D}^T)^{-1}
           \widehat{D}\widehat{M}^{-1}) C^T
   \f]

   The \f$C\f$ matrix is the constraint matrix for enforcing the continuity of
   the "broken" edge space as well as the boundary conditions. This is
   created inside the class.

   Each constraint in turn creates a dual variable (Lagrange multiplier).
   The construction is done locally in each element.
*/
class HybridSolver : public MGLSolver
{
public:
    /**
       @brief Constructor for hybridiziation solver.

       @param mgL Mixed matrices for the graph Laplacian
    */
    template <typename T>
    HybridSolver(const MixedMatrix<T>& mgL);

    virtual ~HybridSolver() = default;

    /// Wrapper for solving the saddle point system through hybridization
    void Solve(const BlockVector& Rhs, BlockVector& Sol) const override;

    /// Transform original RHS to the RHS of the hybridized system
    void RHSTransform(const BlockVector& OriginalRHS, VectorView HybridRHS) const;

    /**
       @brief Recover the solution of the original system from multiplier \f$ \mu \f$.

       \f[
         \left( \begin{array}{c} u \\ p \end{array} \right)
         =
         \left( \begin{array}{c} f \\ g \end{array} \right) -
         \left( \begin{array}{cc} M&  B^T \\ B& \end{array} \right)^{-1}
         \left( \begin{array}{c} C \\ 0 \end{array} \right)
         \mu
       \f]

       This procedure is done locally in each element

       This function assumes the offsets of RecoveredSol have been defined
    */
    void RecoverOriginalSolution(const VectorView& HybridSol,
                                 BlockVector& RecoveredSol) const;

    /**
       @brief Update weights of local M matrices on aggregates
       @param agg_weights weights per aggregate

       @todo when W is non-zero, Aloc and Hybrid_el need to be recomputed
    */
    void UpdateAggScaling(const std::vector<double>& agg_weight);

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) override;
    virtual void SetMaxIter(int max_num_iter) override;
    virtual void SetRelTol(double rtol) override;
    virtual void SetAbsTol(double atol) override;
    ///@}

private:

    template <typename T>
    SparseMatrix AssembleHybridSystem(
        const MixedMatrix<T>& mgl,
        const std::vector<int>& j_multiplier_edgedof);

    SparseMatrix MakeEdgeDofMultiplier() const;

    SparseMatrix MakeLocalC(int agg, const ParMatrix& edge_true_edge,
                            const std::vector<int>& j_multiplier_edgedof,
                            std::vector<int>& edge_map,
                            std::vector<bool>& edge_marker) const;

    void InitSolver(SparseMatrix local_hybrid);

    MPI_Comm comm_;
    int myid_;

    SparseMatrix agg_vertexdof_;
    SparseMatrix agg_edgedof_;
    SparseMatrix agg_multiplier_;

    int num_aggs_;
    int num_edge_dofs_;
    int num_multiplier_dofs_;

    ParMatrix multiplier_d_td_;

    ParMatrix pHybridSystem_;

    linalgcpp::PCGSolver cg_;
    parlinalgcpp::BoomerAMG prec_;

    std::vector<DenseMatrix> MinvDT_;
    std::vector<DenseMatrix> MinvCT_;
    std::vector<DenseMatrix> AinvDMinvCT_;
    std::vector<DenseMatrix> Ainv_;
    std::vector<DenseMatrix> hybrid_elem_;

    mutable std::vector<Vector> Ainv_f_;

    std::vector<double> agg_weights_;

    mutable Vector trueHrhs_;
    mutable Vector trueMu_;
    mutable Vector Hrhs_;
    mutable Vector Mu_;

    bool use_w_;
};

template <typename T>
HybridSolver::HybridSolver(const MixedMatrix<T>& mgl)
    :
    MGLSolver(mgl.Offsets()), comm_(mgl.GlobalD().GetComm()), myid_(mgl.GlobalD().GetMyId()),
    agg_vertexdof_(mgl.agg_vertexdof_),
    agg_edgedof_(mgl.agg_edgedof_),
    num_aggs_(agg_edgedof_.Rows()),
    num_edge_dofs_(agg_edgedof_.Cols()),
    num_multiplier_dofs_(mgl.num_multiplier_dofs_),
    MinvDT_(num_aggs_), MinvCT_(num_aggs_),
    AinvDMinvCT_(num_aggs_), Ainv_(num_aggs_),
    hybrid_elem_(num_aggs_), Ainv_f_(num_aggs_),
    agg_weights_(num_aggs_, 1.0), use_w_(mgl.CheckW())
{
    SparseMatrix edgedof_multiplier = MakeEdgeDofMultiplier();
    SparseMatrix multiplier_edgedof = edgedof_multiplier.Transpose();
    const std::vector<int>& j_multiplier_edgedof = multiplier_edgedof.GetIndices();

    agg_multiplier_ = agg_edgedof_.Mult(edgedof_multiplier);

    ParMatrix edge_td_d = mgl.EdgeTrueEdge().Transpose();
    ParMatrix edge_edge = mgl.EdgeTrueEdge().Mult(edge_td_d);
    ParMatrix edgedof_multiplier_d(comm_, std::move(edgedof_multiplier));
    ParMatrix multiplier_d_td_d = parlinalgcpp::RAP(edge_edge, edgedof_multiplier_d);

    multiplier_d_td_ = MakeEntityTrueEntity(multiplier_d_td_d);

    SparseMatrix local_hybrid = AssembleHybridSystem(mgl, j_multiplier_edgedof);

    InitSolver(std::move(local_hybrid));
}

/// Helper function for assembly
inline
void InvertLocal(const std::vector<double>& elem, std::vector<double>& inverse)
{
    int size = elem.size();

    inverse.resize(size);

    for (int i = 0; i < size; ++i)
    {
        assert(elem[i] != 0.0);

        inverse[i] = 1.0 / elem[i];
    }
}

/// Helper function for assembly
inline
void InvertLocal(const DenseMatrix& elem, DenseMatrix& inverse)
{
    elem.Invert(inverse);
}

/// Helper function for assembly
inline
void MultLocal(const std::vector<double>& Minv, const SparseMatrix& DCloc, DenseMatrix& MinvDCT)
{
    auto DCT = DCloc.Transpose();
    DCT.ScaleRows(Minv);

    DCT.ToDense(MinvDCT);
}

/// Helper function for assembly
inline
void MultLocal(const DenseMatrix& Minv, const SparseMatrix& DCloc, DenseMatrix& MinvDCT)
{
    MinvDCT.SetSize(Minv.Cols(), DCloc.Rows());

    DCloc.MultCT(Minv, MinvDCT);
}

template <typename T>
SparseMatrix HybridSolver::AssembleHybridSystem(
    const MixedMatrix<T>& mgl,
    const std::vector<int>& j_multiplier_edgedof)
{
    const std::vector<T>& M_el = mgl.GetElemM();

    const int map_size = std::max(num_edge_dofs_, agg_vertexdof_.Cols());
    std::vector<int> edge_map(map_size, -1);
    std::vector<bool> edge_marker(num_edge_dofs_, true);

    T Mloc_solver;

    DenseMatrix Aloc;
    DenseMatrix Wloc;
    DenseMatrix CMDADMC;
    DenseMatrix DMinvCT;

    CooMatrix hybrid_system(num_multiplier_dofs_);

    for (int agg = 0; agg < num_aggs_; ++agg)
    {
        // Extracting the size and global numbering of local dof
        std::vector<int> local_vertexdof = agg_vertexdof_.GetIndices(agg);
        std::vector<int> local_edgedof = agg_edgedof_.GetIndices(agg);
        std::vector<int> local_multiplier = agg_multiplier_.GetIndices(agg);

        const int nlocal_vertexdof = local_vertexdof.size();
        const int nlocal_multiplier = local_multiplier.size();

        SparseMatrix Dloc = mgl.LocalD().GetSubMatrix(local_vertexdof, local_edgedof,
                                                      edge_map);

        SparseMatrix Cloc = MakeLocalC(agg, mgl.EdgeTrueEdge(), j_multiplier_edgedof,
                                       edge_map, edge_marker);

        // Compute:
        //      CMinvCT = Cloc * MinvCT
        //      Aloc = DMinvDT = Dloc * MinvDT
        //      DMinvCT = Dloc * MinvCT
        //      CMinvDTAinvDMinvCT = CMinvDT * AinvDMinvCT_
        //      hybrid_elem = CMinvCT - CMinvDTAinvDMinvCT

        InvertLocal(M_el[agg], Mloc_solver);

        DenseMatrix& MinvCT_i(MinvCT_[agg]);
        DenseMatrix& MinvDT_i(MinvDT_[agg]);
        DenseMatrix& AinvDMinvCT_i(AinvDMinvCT_[agg]);
        DenseMatrix& Ainv_i(Ainv_[agg]);
        DenseMatrix& hybrid_elem(hybrid_elem_[agg]);

        AinvDMinvCT_i.SetSize(nlocal_vertexdof, nlocal_multiplier);
        hybrid_elem.SetSize(nlocal_multiplier, nlocal_multiplier);
        Aloc.SetSize(nlocal_vertexdof, nlocal_vertexdof);
        DMinvCT.SetSize(nlocal_vertexdof, nlocal_multiplier);

        MultLocal(Mloc_solver, Dloc, MinvDT_i);
        MultLocal(Mloc_solver, Cloc, MinvCT_i);

        Cloc.Mult(MinvCT_i, hybrid_elem);
        Dloc.Mult(MinvCT_i, DMinvCT);
        Dloc.Mult(MinvDT_i, Aloc);

        if (use_w_)
        {
            auto Wloc_tmp = mgl.LocalW().GetSubMatrix(local_vertexdof, local_vertexdof, edge_map);
            Wloc_tmp.ToDense(Wloc);

            Aloc -= Wloc;
        }

        Aloc.Invert(Ainv_i);

        Ainv_i.Mult(DMinvCT, AinvDMinvCT_i);

        if (DMinvCT.Rows() > 0 && DMinvCT.Rows() > 0)
        {
            CMDADMC.SetSize(nlocal_multiplier, nlocal_multiplier);
            AinvDMinvCT_i.MultAT(DMinvCT, CMDADMC);
            hybrid_elem -= CMDADMC;
        }

        // Add contribution of the element matrix to the global system
        hybrid_system.Add(local_multiplier, hybrid_elem);
    }

    return hybrid_system.ToSparse();
}


} // namespace smoothg

#endif /* HYBRIDSOLVER_HPP_ */
