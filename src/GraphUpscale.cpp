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

    @brief Contains GraphUpscale class
*/

#include "GraphUpscale.hpp"

namespace smoothg
{

GraphUpscale::GraphUpscale(const Graph& graph, const UpscaleParams& params)
    : Operator(graph.vertex_edge_local_.Rows()),
      comm_(graph.edge_true_edge_.GetComm()),
      myid_(graph.edge_true_edge_.GetMyId()),
      setup_time_(0),
      hybridization_(params.hybridization)
{
    Timer timer(Timer::Start::True);

    // Compute Topology
    std::vector<GraphTopology> gts;
    gts.emplace_back(graph);

    for (int level = 1; level < params.max_levels - 1; ++level)
    {
        gts.emplace_back(gts.back(), params.coarsen_factor);
    }

    // Fine Level
    {
        levels_.emplace_back(graph, params.elim_edge_dofs);
    }

    // Coarsen Levels
    for (int level_i = 0; level_i < params.max_levels - 1; ++level_i)
    {
        auto& gt_i = gts[level_i];
        const auto& prev_level = GetLevel(level_i);
        const auto& spect_pair_i = params.spectral_pair[level_i];

        coarsener_.emplace_back(std::move(gt_i), prev_level, spect_pair_i);
        levels_.push_back(coarsener_.back().Coarsen(prev_level));
    }

    // Generate Solvers (potentially optional)
    for (int level_i = 0; level_i < NumLevels(); ++level_i)
    {
        MakeSolver(level_i);
    }

    SetOrthogonalize(!GetMatrix(0).CheckW());

    timer.Click();
    setup_time_ += timer.TotalTime();
}

void GraphUpscale::MakeSolver(int level_i)
{
    MakeSolver(level_i, GetMatrix(level_i));
}

void GraphUpscale::MakeSolver(int level_i, MixedMatrix& mm)
{
    auto& level = GetLevel(level_i);

    if (level_i == 0)
    {
        mm.AssembleM();
        level.solver = make_unique<SPDSolver>(mm, level.elim_dofs);
    }
    else if (hybridization_)
    {
        level.solver = make_unique<HybridSolver>(mm, GetGraphSpace(level_i));
    }
    else
    {
        mm.AssembleM();
        level.solver = make_unique<MinresBlockSolver>(mm, level.elim_dofs);
    }

    size_to_level_[mm.LocalD().Rows()] = level_i;
}

void GraphUpscale::RescaleSolver(int level_i, const std::vector<double>& agg_weights)
{
    RescaleSolver(level_i, agg_weights, GetMatrix(level_i));
}

void GraphUpscale::RescaleSolver(int level_i, const std::vector<double>& agg_weights,
                                 MixedMatrix& mm)
{
    auto& level = GetLevel(level_i);

    if (level_i == 0)
    {
        mm.AssembleM(agg_weights);

        level.solver = make_unique<SPDSolver>(mm, level.elim_dofs);
    }
    else if (hybridization_)
    {
        if (!level.solver)
        {
            level.solver = make_unique<HybridSolver>(mm, GetGraphSpace(level_i));
        }

        auto& hb = dynamic_cast<HybridSolver&>(*level.solver);
        hb.UpdateAggScaling(agg_weights);
    }
    else
    {
        mm.AssembleM(agg_weights);
        level.solver = make_unique<MinresBlockSolver>(mm, level.elim_dofs);
    }

    size_to_level_[mm.LocalD().Rows()] = level_i;
}

std::vector<BlockVector> GraphUpscale::MultMultiLevel(const BlockVector& x) const
{
    std::vector<BlockVector> sols;

    MultMultiLevel(x, sols);

    return sols;
}

void GraphUpscale::MultMultiLevel(const BlockVector& x, std::vector<BlockVector>& sols) const
{
    int num_levels = NumLevels();

    sols.resize(num_levels, GetBlockVector(0));

    for (int i = 0; i < num_levels; ++i)
    {
        Solve(i, x, sols[i]);
    }
}

BlockVector GraphUpscale::MultMultiGrid(const BlockVector& x) const
{
    BlockVector sol = GetBlockVector(0);
    sol = 0.0;

    MultMultiGrid(x, sol);

    return sol;
}

void GraphUpscale::MultMultiGrid(const BlockVector& x, BlockVector& sol) const
{
    GetLevel(0).rhs = x;
    GetLevel(0).sol = sol;

    int num_coarse = NumLevels() - 1;

    for (int i = 0; i < num_coarse; ++i)
    {
        Coarsener(i).Restrict(GetLevel(i).rhs, GetLevel(i + 1).rhs);
        Coarsener(i).Restrict(GetLevel(i).sol, GetLevel(i + 1).sol);
    }

    int num_levels = NumLevels();

    for (int i = num_levels - 1; i >= 0; --i)
    {
        GetLevel(i).rhs.GetBlock(1) *= -1.0;
        GetLevel(i).sol.GetBlock(1) *= -1.0;

        Solver(i).Solve(GetLevel(i).rhs, GetLevel(i).sol);

        if (do_ortho_)
        {
            OrthoConstant(comm_, GetLevel(i).sol.GetBlock(1), ConstantRep(i));
        }

        if (i != 0)
        {
            Coarsener(i - 1).Interpolate(GetLevel(i).sol, GetLevel(i - 1).sol);
        }
    }

    sol = GetLevel(0).sol;

    if (do_ortho_)
    {
        OrthoConstant(comm_, sol.GetBlock(1), ConstantRep(0));
    }
}

void GraphUpscale::Mult(const VectorView& x, VectorView y) const
{
    Solve(1, x, y);
}

void GraphUpscale::Solve(const VectorView& x, VectorView y) const
{
    Solve(1, x, y);
}

Vector GraphUpscale::Solve(const VectorView& x) const
{
    return Solve(1, x);
}

void GraphUpscale::Solve(const BlockVector& x, BlockVector& y) const
{
    Solve(1, x, y);
}

BlockVector GraphUpscale::Solve(const BlockVector& x) const
{
    return Solve(1, x);
}

void GraphUpscale::Solve(int level, const VectorView& x, VectorView y) const
{
    GetLevel(0).rhs.GetBlock(0) = 0.0;
    GetLevel(0).rhs.GetBlock(1) = x;

    for (int i = 0; i < level; ++i)
    {
        Coarsener(i).Restrict(GetLevel(i).rhs, GetLevel(i + 1).rhs);
    }

    GetLevel(level).rhs.GetBlock(1) *= -1.0;
    GetLevel(level).sol = 0.0;

    Solver(level).Solve(GetLevel(level).rhs, GetLevel(level).sol);

    if (do_ortho_)
    {
        Orthogonalize(level, GetLevel(level).sol.GetBlock(1));
    }

    for (int i = level - 1; i >= 0; --i)
    {
        Coarsener(i).Interpolate(GetLevel(i + 1).sol, GetLevel(i).sol);
    }

    y = GetLevel(0).sol.GetBlock(1);
}

Vector GraphUpscale::Solve(int level, const VectorView& x) const
{
    Vector y = GetVector(0);

    Solve(level, x, y);

    return y;
}

void GraphUpscale::Solve(int level, const BlockVector& x, BlockVector& y) const
{
    GetLevel(0).rhs = x;

    for (int i = 0; i < level; ++i)
    {
        Coarsener(i).Restrict(GetLevel(i).rhs, GetLevel(i + 1).rhs);
    }

    GetLevel(level).rhs.GetBlock(1) *= -1.0;
    GetLevel(level).sol = 0.0;

    Solver(level).Solve(GetLevel(level).rhs, GetLevel(level).sol);

    if (do_ortho_)
    {
        Orthogonalize(level, GetLevel(level).sol);
    }

    for (int i = level - 1; i >= 0; --i)
    {
        Coarsener(i).Interpolate(GetLevel(i + 1).sol, GetLevel(i).sol);
    }

    y = GetLevel(0).sol;
}

BlockVector GraphUpscale::Solve(int level, const BlockVector& x) const
{
    BlockVector y = GetBlockVector(0);

    Solve(level, x, y);

    return y;
}

void GraphUpscale::SolveLevel(int level, const VectorView& x, VectorView y) const
{
    Solver(level).Solve(x, y);
    y *= -1.0;

    if (do_ortho_)
    {
        Orthogonalize(level, y);
    }
}

Vector GraphUpscale::SolveLevel(int level, const VectorView& x) const
{
    Vector y = GetVector(level);
    SolveLevel(level, x, y);

    return y;
}

void GraphUpscale::SolveLevel(int level, const BlockVector& x, BlockVector& y) const
{
    Solver(level).Solve(x, y);
    y.GetBlock(1) *= -1.0;

    if (do_ortho_)
    {
        Orthogonalize(level, y);
    }
}

BlockVector GraphUpscale::SolveLevel(int level, const BlockVector& x) const
{
    BlockVector y = GetBlockVector(level);
    SolveLevel(level, x, y);

    return y;
}

void GraphUpscale::Interpolate(const VectorView& x, VectorView y) const
{
    int x_level = size_to_level_.at(x.size());
    int y_level = size_to_level_.at(y.size());

    // Don't need temp space and copy if only between consecutive levels
    if (x_level - y_level == 1)
    {
        Coarsener(y_level).Interpolate(x, y);

        return;
    }

    GetLevel(x_level).sol.GetBlock(1) = x;

    for (int i = x_level - 1; i >= y_level; --i)
    {
        Coarsener(i).Interpolate(GetLevel(i + 1).sol.GetBlock(1), GetLevel(i).sol.GetBlock(1));
    }

    y = GetLevel(y_level).sol.GetBlock(1);
}

Vector GraphUpscale::Interpolate(const VectorView& x, int level) const
{
    Vector y = GetVector(level);

    Interpolate(x, y);

    return y;
}

void GraphUpscale::Interpolate(const BlockVector& x, BlockVector& y) const
{
    int x_level = size_to_level_.at(x.GetBlock(1).size());
    int y_level = size_to_level_.at(y.GetBlock(1).size());

    // Don't need temp space and copy if only between consecutive levels
    if (x_level - y_level == 1)
    {
        Coarsener(y_level).Interpolate(x, y);

        return;
    }

    GetLevel(x_level).sol = x;

    for (int i = x_level - 1; i >= y_level; --i)
    {
        Coarsener(i).Interpolate(GetLevel(i + 1).sol, GetLevel(i).sol);
    }

    y = GetLevel(y_level).sol;
}

BlockVector GraphUpscale::Interpolate(const BlockVector& x, int level) const
{
    BlockVector y = GetBlockVector(level);

    Interpolate(x, y);

    return y;
}

void GraphUpscale::Restrict(const VectorView& x, VectorView y) const
{
    int x_level = size_to_level_.at(x.size());
    int y_level = size_to_level_.at(y.size());

    // Don't need temp space and copy if only between consecutive levels
    if (y_level - x_level == 1)
    {
        Coarsener(x_level).Restrict(x, y);

        return;
    }

    GetLevel(x_level).sol.GetBlock(1) = x;

    for (int i = x_level; i < y_level; ++i)
    {
        Coarsener(i).Restrict(GetLevel(i).sol.GetBlock(1), GetLevel(i + 1).sol.GetBlock(1));
    }

    y = GetLevel(y_level).sol.GetBlock(1);
}

Vector GraphUpscale::Restrict(const VectorView& x, int level) const
{
    Vector y = GetVector(level);

    Restrict(x, y);

    return y;
}

void GraphUpscale::Restrict(const BlockVector& x, BlockVector& y) const
{
    int x_level = size_to_level_.at(x.GetBlock(1).size());
    int y_level = size_to_level_.at(y.GetBlock(1).size());

    // Don't need temp space and copy if only between consecutive levels
    if (y_level - x_level == 1)
    {
        Coarsener(x_level).Restrict(x, y);

        return;
    }

    GetLevel(x_level).sol = x;

    for (int i = x_level; i < y_level; ++i)
    {
        Coarsener(i).Restrict(GetLevel(i).sol, GetLevel(i + 1).sol);
    }

    y = GetLevel(y_level).sol;
}

BlockVector GraphUpscale::Restrict(const BlockVector& x, int level) const
{
    BlockVector y = GetBlockVector(level);

    Restrict(x, y);

    return y;
}

void GraphUpscale::Project(const BlockVector& x, BlockVector& y) const
{
    Coarsener(0).Project(x, y);
}

BlockVector GraphUpscale::Project(const BlockVector& x) const
{
    return Coarsener(0).Project(x);
}

const std::vector<int>& GraphUpscale::BlockOffsets(int level) const
{
    return GetMatrix(level).Offsets();
}

const std::vector<int>& GraphUpscale::TrueBlockOffsets(int level) const
{
    return GetMatrix(level).TrueOffsets();
}

void GraphUpscale::Orthogonalize(int level, VectorView vect) const
{
    OrthoConstant(comm_, vect, ConstantRep(level));
}

void GraphUpscale::Orthogonalize(int level, BlockVector& vect) const
{
    Orthogonalize(level, vect.GetBlock(1));
}

Vector GraphUpscale::GetVector(int level) const
{
    return Vector(GetMatrix(level).LocalD().Rows());
}

BlockVector GraphUpscale::GetBlockVector(int level) const
{
    return BlockVector(GetMatrix(level).Offsets());
}

BlockVector GraphUpscale::GetTrueBlockVector(int level) const
{
    return BlockVector(GetMatrix(level).TrueOffsets());
}

std::vector<Vector> GraphUpscale::GetMLVectors() const
{
    int num_levels = NumLevels();

    std::vector<Vector> vects(num_levels);

    for (int i = 0; i < num_levels; ++i)
    {
        vects[i] = GetVector(i);
    }

    return vects;
}

std::vector<BlockVector> GraphUpscale::GetMLBlockVector() const
{
    int num_levels = NumLevels();

    std::vector<BlockVector> vects(num_levels);

    for (int i = 0; i < num_levels; ++i)
    {
        vects[i] = GetBlockVector(i);
    }

    return vects;
}

MixedMatrix& GraphUpscale::GetMatrix(int level)
{
    return GetLevel(level).mixed_matrix;
}

const MixedMatrix& GraphUpscale::GetMatrix(int level) const
{
    return GetLevel(level).mixed_matrix;
}

int GraphUpscale::GlobalRows() const
{
    return GetMatrix(0).GlobalD().GlobalRows();
}

int GraphUpscale::GlobalCols() const
{
    return GetMatrix(0).GlobalD().GlobalRows();
}

void GraphUpscale::PrintInfo(std::ostream& out) const
{
    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    if (myid_ == 0)
    {
        int old_precision = out.precision();
        out.precision(3);

        out << "\n";

        if (num_procs > 1)
        {
            out << "Processors: " << num_procs << "\n";
            out << "---------------------\n";
        }

        out << "\n";

        for (int i = 0; i < NumLevels(); ++i)
        {
            out << "Level " << i << " Matrix\n";
            out << "---------------------\n";
            out << "M Size\t\t" << GetMatrix(i).GlobalM().GlobalRows() << "\n";
            out << "D Size\t\t" << GetMatrix(i).GlobalD().GlobalRows() << "\n";
            out << "+ Size\t\t" << GetMatrix(i).GlobalRows() << "\n";
            out << "NonZeros:\t" << GetMatrix(i).GlobalNNZ() << "\n";
            out << "\n";

            if (i != 0)
            {
                double op_comp = 1.0 + (Solver(i).GetNNZ() / (double) Solver(0).GetNNZ());

                out << "Op Comp:\t" << op_comp << "\n";
                out << "\n";
            }
        }

        out.precision(old_precision);
    }
}

double GraphUpscale::OperatorComplexity() const
{
    double nnz_levels = 0.0;

    for (auto&& level : levels_)
    {
        nnz_levels += level.solver->GetNNZ();
    }

    double nnz_fine;

    if (GetLevel(0).solver)
    {
        nnz_fine = GetLevel(0).solver->GetNNZ();
    }
    else
    {
        nnz_fine = GetMatrix(0).GlobalNNZ();
    }

    return nnz_levels / nnz_fine;
}

void GraphUpscale::SetPrintLevel(int print_level)
{
    for (auto&& level : levels_)
    {
        if (level.solver)
        {
            level.solver->SetPrintLevel(print_level);
        }
    }
}

void GraphUpscale::SetMaxIter(int max_num_iter)
{
    for (auto&& level : levels_)
    {
        if (level.solver)
        {
            level.solver->SetMaxIter(max_num_iter);
        }
    }
}

void GraphUpscale::SetRelTol(double rtol)
{
    for (auto& level : levels_)
    {
        if (level.solver)
        {
            level.solver->SetRelTol(rtol);
        }
    }
}

void GraphUpscale::SetAbsTol(double atol)
{
    for (auto& level : levels_)
    {
        if (level.solver)
        {
            level.solver->SetAbsTol(atol);
        }
    }
}

void GraphUpscale::ShowCoarseSolveInfo(std::ostream& out) const
{
    if (myid_ == 0)
    {
        for (int i = 1; i < NumLevels(); ++i)
        {
            out << "\n";
            out << "Level " << i << " Solve Time:       " << Solver(i).GetTiming() << "\n";
            out << "Level " << i << " Solve Iterations: " << Solver(i).GetNumIterations() << "\n";
        }
    }
}

void GraphUpscale::ShowFineSolveInfo(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "Fine Solve Time:         " << Solver(0).GetTiming() << "\n";
        out << "Fine Solve Iterations:   " << Solver(0).GetNumIterations() << "\n";
    }
}

void GraphUpscale::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "GraphUpscale Setup Time:      " << setup_time_ << "\n";
    }
}

double GraphUpscale::SolveTime(int level) const
{
    return Solver(level).GetTiming();
}

int GraphUpscale::SolveIters(int level) const
{
    return Solver(level).GetNumIterations();
}

double GraphUpscale::GetSetupTime() const
{
    return setup_time_;
}

std::vector<double> GraphUpscale::ComputeErrors(const BlockVector& upscaled_sol,
                                                const BlockVector& fine_sol) const
{
    const SparseMatrix& M = GetMatrix(0).LocalM();
    const SparseMatrix& D = GetMatrix(0).LocalD();

    auto info = smoothg::ComputeErrors(comm_, M, D, upscaled_sol, fine_sol);
    info.push_back(OperatorComplexity());

    return info;
}

void GraphUpscale::ShowErrors(const BlockVector& upscaled_sol,
                              const BlockVector& fine_sol) const
{
    auto info = ComputeErrors(upscaled_sol, fine_sol);

    if (myid_ == 0)
    {
        smoothg::ShowErrors(info);
    }
}

ParMatrix GraphUpscale::ToPrimal() const
{
    const auto& mgl = GetMatrix(0);

    std::vector<double> M_diag(mgl.GlobalM().GetDiag().GetDiag());

    SparseMatrix D_elim = mgl.LocalD();

    bool use_w = mgl.CheckW();

    if (GetLevel(0).elim_dofs.size() > 0)
    {
        std::vector<int> marker(D_elim.Cols(), 0);

        for (auto&& dof : GetLevel(0).elim_dofs)
        {
            marker[dof] = 1;
        }

        D_elim.EliminateCol(marker);
    }

    ParMatrix D_elim_global(comm_, mgl.GlobalD().GetRowStarts(),
                            mgl.EdgeTrueEdge().GetRowStarts(), std::move(D_elim));

    ParMatrix D = D_elim_global.Mult(mgl.EdgeTrueEdge());
    ParMatrix MinvDT = D.Transpose();
    MinvDT.InverseScaleRows(M_diag);

    ParMatrix A;

    if (use_w)
    {
        A = parlinalgcpp::ParSub(D.Mult(MinvDT), mgl.GlobalW());
    }
    else
    {
        A = D.Mult(MinvDT);
    }

    return A;
}

} // namespace smoothg
