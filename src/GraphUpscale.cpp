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

GraphUpscale::GraphUpscale(Graph graph, double spect_tol, int max_evects, bool hybridization,
                           int num_levels, const std::vector<int>& edge_elim_dofs)
    : Operator(graph.vertex_edge_local_.Rows()),
      comm_(graph.edge_true_edge_.GetComm()),
      myid_(graph.edge_true_edge_.GetMyId()),
      global_vertices_(graph.global_vertices_),
      global_edges_(graph.global_edges_),
      setup_time_(0),
      spect_tol_(spect_tol), max_evects_(max_evects),
      hybridization_(hybridization),
      graph_(std::move(graph))
{
    Timer timer(Timer::Start::True);

    double coarsen_factor = 4.0;

    std::vector<GraphTopology> gts;

    BlockVector elim_vect;

    // Fine Level
    {
        gts.emplace_back(graph_);

        mgl_.emplace_back(graph_);
        mgl_.back().AssembleM();
        constant_rep_.emplace_back(Rows(), 1.0 / std::sqrt(GlobalRows()));

        elim_dofs_.push_back(edge_elim_dofs);

        elim_vect = GetFineBlockVector();
        elim_vect = 0.0;

        for (auto&& dof : elim_dofs_.back())
        {
            elim_vect.GetBlock(0)[dof] = 10.0;
        }

        rhs_.emplace_back(mgl_.back().Offsets());
        sol_.emplace_back(mgl_.back().Offsets());

        solver_.push_back(make_unique<SPDSolver>(mgl_.back(), elim_dofs_.back()));
    }

    // Coarse Levels
    for (int i = 1; i < num_levels; ++i)
    {
        ParPrint(myid_, printf("Coarsening: %d -> %d\n", gts.back().agg_vertex_local_.Cols(),
                    gts.back().agg_vertex_local_.Rows()));
        //coarsener_.emplace_back(gts.back(), mgl_.back(), max_evects_, spect_tol_);
        coarsener_.emplace_back(gts.back(), mgl_.back(), i + max_evects_, spect_tol_);
        //coarsener_.emplace_back(gts.back(), mgl_.back(), 5 + i + max_evects_, spect_tol_);
        //coarsener_.emplace_back(gts.back(), mgl_.back(), (i / 2) + max_evects_, spect_tol_);
        mgl_.push_back(coarsener_.back().Coarsen(mgl_.back()));

        rhs_.emplace_back(mgl_.back().Offsets());
        sol_.emplace_back(mgl_.back().Offsets());

        constant_rep_.emplace_back(coarsener_.back().Restrict(constant_rep_.back()));

        elim_vect = coarsener_.back().Restrict(elim_vect);
        elim_dofs_.emplace_back(std::vector<int>());

        {
            int ml_num_edges = elim_vect.GetBlock(0).size();

            for (int i = 0; i < ml_num_edges; ++i)
            {
                if (std::fabs(elim_vect[i]) > 1e-14)
                {
                    elim_dofs_.back().push_back(i);
                }
            }
        }

        solver_.push_back(make_unique<MinresBlockSolver>(mgl_.back(), elim_dofs_.back()));

        if (i != num_levels - 1)
        {
            gts.emplace_back(gts.back(), coarsen_factor);
            mgl_.back().AssembleM(); // Coarsening requires assembled M for now
        }
    }

    do_ortho_ = !GetFineMatrix().CheckW();

    timer.Click();
    setup_time_ += timer.TotalTime();
}

void GraphUpscale::MakeCoarseSolver()
{
    auto& mm = GetCoarseMatrix();

    if (hybridization_)
    {
        solver_[1] = make_unique<HybridSolver>(mm);
    }
    else
    {
        mm.AssembleM();
        //coarse_solver_ = make_unique<MinresBlockSolver>(mm, coarse_elim_dofs_);
        solver_[1] = make_unique<MinresBlockSolver>(mm, elim_dofs_[1]);
    }
}

void GraphUpscale::MakeFineSolver()
{
    auto& mm = GetFineMatrix();

    mm.AssembleM();
    solver_[0] = make_unique<SPDSolver>(mm, elim_dofs_[0]);
}

void GraphUpscale::MakeCoarseSolver(const std::vector<double>& agg_weights)
{
    auto& mm = GetCoarseMatrix();

    if (hybridization_)
    {
        if (!solver_[1])
        {
            MakeCoarseSolver();
        }

        auto& hb = dynamic_cast<HybridSolver&>(*solver_[1]);
        hb.UpdateAggScaling(agg_weights);
    }
    else
    {
        mm.AssembleM(agg_weights);
        solver_[1] = make_unique<MinresBlockSolver>(mm, elim_dofs_[1]);
    }
}

void GraphUpscale::MakeFineSolver(const std::vector<double>& agg_weights)
{
    auto& mm = GetFineMatrix();

    mm.AssembleM(agg_weights);
    solver_[0] = make_unique<SPDSolver>(mm, elim_dofs_[0]);
}

Vector GraphUpscale::ReadVertexVector(const std::string& filename) const
{
    return ReadVector(filename, graph_.vertex_map_);
}

Vector GraphUpscale::ReadEdgeVector(const std::string& filename) const
{
    return ReadVector(filename, graph_.edge_map_);
}

BlockVector GraphUpscale::ReadVertexBlockVector(const std::string& filename) const
{
    BlockVector vect = GetFineBlockVector();

    vect.GetBlock(0) = 0.0;
    vect.GetBlock(1) = ReadVertexVector(filename);

    return vect;
}

BlockVector GraphUpscale::ReadEdgeBlockVector(const std::string& filename) const
{
    BlockVector vect = GetFineBlockVector();

    vect.GetBlock(0) = ReadEdgeVector(filename);
    vect.GetBlock(1) = 0.0;

    return vect;
}

std::vector<BlockVector> GraphUpscale::MultMultiLevel(const BlockVector& x) const
{
    int num_levels = solver_.size();

    std::vector<BlockVector> sols(num_levels);

    MultMultiLevel(x, sols);

    return sols;
}

void GraphUpscale::MultMultiLevel(const BlockVector& x, std::vector<BlockVector>& sols) const
{
    rhs_[0] = x;

    int num_coarse = coarsener_.size();

    for (int i = 0; i < num_coarse; ++i)
    {
        coarsener_[i].Restrict(rhs_[i], rhs_[i + 1]);
    }

    int num_levels = solver_.size();

    for (int i = 0; i < num_levels; ++i)
    {
        rhs_[i].GetBlock(1) *= -1.0;
        solver_[i]->Solve(rhs_[i], sol_[i]);

        if (do_ortho_)
        {
            OrthoConstant(comm_, sol_[i].GetBlock(1), constant_rep_[i]);
        }
    }

    sols.resize(num_levels);

    for (int i = 0; i < num_levels; ++i)
    {
        for (int j = i; j > 0; --j)
        {
            coarsener_[j - 1].Interpolate(sol_[j], sol_[j - 1]);
        }

        sols[i] = sol_[0];
    }
}

void GraphUpscale::Mult(const VectorView& x, VectorView y) const
{
    coarsener_[0].Restrict(x, rhs_[1].GetBlock(1));

    rhs_[1].GetBlock(0) = 0.0;
    rhs_[1].GetBlock(1) *= -1.0;

    solver_[1]->Solve(rhs_[1], sol_[1]);

    if (do_ortho_)
    {
        OrthoConstant(comm_, sol_[1].GetBlock(1), constant_rep_[1]);
    }

    coarsener_[0].Interpolate(sol_[1].GetBlock(1), y);
}

void GraphUpscale::Solve(const VectorView& x, VectorView y) const
{
    Mult(x, y);
}

Vector GraphUpscale::Solve(const VectorView& x) const
{
    Vector y(x.size());

    Solve(x, y);

    return y;
}

void GraphUpscale::Solve(const BlockVector& x, BlockVector& y) const
{
    coarsener_[0].Restrict(x, rhs_[1]);
    rhs_[1].GetBlock(1) *= -1.0;

    solver_[1]->Solve(rhs_[1], sol_[1]);

    if (do_ortho_)
    {
        OrthoConstant(comm_, sol_[1].GetBlock(1), constant_rep_[1]);
    }

    coarsener_[0].Interpolate(sol_[1], y);
}

BlockVector GraphUpscale::Solve(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    Solve(x, y);

    return y;
}

void GraphUpscale::SolveCoarse(const VectorView& x, VectorView y) const
{
    solver_[1]->Solve(x, y);
    y *= -1.0;

    if (do_ortho_)
    {
        OrthoConstant(comm_, y, constant_rep_[1]);
    }
}

Vector GraphUpscale::SolveCoarse(const VectorView& x) const
{
    Vector coarse_vect = GetCoarseVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void GraphUpscale::SolveCoarse(const BlockVector& x, BlockVector& y) const
{
    solver_[1]->Solve(x, y);
    y *= -1.0;

    if (do_ortho_)
    {
        OrthoConstant(comm_, y.GetBlock(1), constant_rep_[1]);
    }
}

BlockVector GraphUpscale::SolveCoarse(const BlockVector& x) const
{
    BlockVector coarse_vect = GetCoarseBlockVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void GraphUpscale::SolveFine(const VectorView& x, VectorView y) const
{
    assert(solver_[0]);

    solver_[0]->Solve(x, y);
    y *= -1.0;

    if (do_ortho_)
    {
        OrthoConstant(comm_, y, constant_rep_[0]);
    }
}

Vector GraphUpscale::SolveFine(const VectorView& x) const
{
    Vector y(x.size());

    SolveFine(x, y);

    return y;
}

void GraphUpscale::SolveFine(const BlockVector& x, BlockVector& y) const
{
    assert(solver_[0]);

    solver_[0]->Solve(x, y);
    y *= -1.0;

    if (do_ortho_)
    {
        OrthoConstant(comm_, y.GetBlock(1), constant_rep_[0]);
    }
}

BlockVector GraphUpscale::SolveFine(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    SolveFine(x, y);

    return y;
}

void GraphUpscale::Interpolate(const VectorView& x, VectorView y) const
{
    coarsener_[0].Interpolate(x, y);
}

Vector GraphUpscale::Interpolate(const VectorView& x) const
{
    return coarsener_[0].Interpolate(x);
}

void GraphUpscale::Interpolate(const BlockVector& x, BlockVector& y) const
{
    coarsener_[0].Interpolate(x, y);
}

BlockVector GraphUpscale::Interpolate(const BlockVector& x) const
{
    return coarsener_[0].Interpolate(x);
}

void GraphUpscale::Restrict(const VectorView& x, VectorView y) const
{
    coarsener_[0].Restrict(x, y);
}

Vector GraphUpscale::Restrict(const VectorView& x) const
{
    return coarsener_[0].Restrict(x);
}

void GraphUpscale::Restrict(const BlockVector& x, BlockVector& y) const
{
    coarsener_[0].Restrict(x, y);
}

BlockVector GraphUpscale::Restrict(const BlockVector& x) const
{
    return coarsener_[0].Restrict(x);
}

const std::vector<int>& GraphUpscale::FineBlockOffsets() const
{
    return GetFineMatrix().Offsets();
}

const std::vector<int>& GraphUpscale::CoarseBlockOffsets() const
{
    return GetCoarseMatrix().Offsets();
}

const std::vector<int>& GraphUpscale::FineTrueBlockOffsets() const
{
    return GetFineMatrix().TrueOffsets();
}

const std::vector<int>& GraphUpscale::CoarseTrueBlockOffsets() const
{
    return GetCoarseMatrix().TrueOffsets();
}

void GraphUpscale::Orthogonalize(VectorView vect) const
{
    OrthoConstant(comm_, vect, GlobalRows());
}

void GraphUpscale::Orthogonalize(BlockVector& vect) const
{
    Orthogonalize(vect.GetBlock(1));
}

void GraphUpscale::OrthogonalizeCoarse(VectorView vect) const
{
    OrthoConstant(comm_, vect, GetCoarseConstant());
}

void GraphUpscale::OrthogonalizeCoarse(BlockVector& vect) const
{
    OrthogonalizeCoarse(vect.GetBlock(1));
}

const Vector& GraphUpscale::GetCoarseConstant() const
{
    return constant_rep_[1];
}

Vector GraphUpscale::GetCoarseVector() const
{
    int coarse_size = GetCoarseMatrix().LocalD().Rows();

    return Vector(coarse_size);
}

Vector GraphUpscale::GetFineVector() const
{
    int fine_size = GetFineMatrix().LocalD().Rows();

    return Vector(fine_size);
}

BlockVector GraphUpscale::GetCoarseBlockVector() const
{
    return BlockVector(GetCoarseMatrix().Offsets());
}

BlockVector GraphUpscale::GetFineBlockVector() const
{
    return BlockVector(GetFineMatrix().Offsets());
}

BlockVector GraphUpscale::GetCoarseTrueBlockVector() const
{
    return BlockVector(GetCoarseMatrix().TrueOffsets());
}

BlockVector GraphUpscale::GetFineTrueBlockVector() const
{
    return BlockVector(GetFineMatrix().TrueOffsets());
}

MixedMatrix& GraphUpscale::GetFineMatrix()
{
    return GetMatrix(0);
}

const MixedMatrix& GraphUpscale::GetFineMatrix() const
{
    return GetMatrix(0);
}

MixedMatrix& GraphUpscale::GetCoarseMatrix()
{
    return GetMatrix(1);
}

const MixedMatrix& GraphUpscale::GetCoarseMatrix() const
{
    return GetMatrix(1);
}

MixedMatrix& GraphUpscale::GetMatrix(int level)
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));

    return mgl_[level];
}

const MixedMatrix& GraphUpscale::GetMatrix(int level) const
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));

    return mgl_[level];
}

int GraphUpscale::GlobalRows() const
{
    return GetFineMatrix().GlobalD().GlobalRows();
}

int GraphUpscale::GlobalCols() const
{
    return GetFineMatrix().GlobalD().GlobalRows();
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

        for (size_t i = 0; i < mgl_.size(); ++i)
        {
            out << "Level " << i << " Matrix\n";
            out << "---------------------\n";
            out << "M Size\t\t" << mgl_[i].GlobalM().GlobalRows() << "\n";
            out << "D Size\t\t" << mgl_[i].GlobalD().GlobalRows() << "\n";
            out << "  Size\t\t" << mgl_[i].GlobalRows() << "\n";
            out << "NonZeros:\t" << mgl_[i].GlobalNNZ() << "\n";
            out << "\n";

            if (i != 0)
            {
                double op_comp = 1.0 + (solver_[i]->GetNNZ() / (double) solver_[0]->GetNNZ());

                out << "Op Comp:\t" << op_comp << "\n";
                out << "\n";
            }
        }

        out.precision(old_precision);
    }
}

double GraphUpscale::OperatorComplexity() const
{
    assert(solver_[1]);

    double nnz_coarse = 0.0;

    for (auto&& solver : solver_)
    {
        nnz_coarse += solver->GetNNZ();
    }

    double nnz_fine;

    if (solver_[0])
    {
        nnz_fine = solver_[0]->GetNNZ();
    }
    else
    {
        nnz_fine = GetFineMatrix().GlobalNNZ();
    }

    double op_comp = nnz_coarse / nnz_fine;

    return op_comp;
}

void GraphUpscale::SetPrintLevel(int print_level)
{
    assert(solver_[1]);
    solver_[1]->SetPrintLevel(print_level);

    if (solver_[0])
    {
        solver_[0]->SetPrintLevel(print_level);
    }
}

void GraphUpscale::SetMaxIter(int max_num_iter)
{
    assert(solver_[1]);
    solver_[1]->SetMaxIter(max_num_iter);

    if (solver_[0])
    {
        solver_[0]->SetMaxIter(max_num_iter);
    }
}

void GraphUpscale::SetRelTol(double rtol)
{
    assert(solver_[1]);
    solver_[1]->SetRelTol(rtol);

    if (solver_[0])
    {
        solver_[0]->SetRelTol(rtol);
    }
}

void GraphUpscale::SetAbsTol(double atol)
{
    assert(solver_[1]);
    solver_[1]->SetAbsTol(atol);

    if (solver_[0])
    {
        solver_[0]->SetAbsTol(atol);
    }
}

void GraphUpscale::ShowCoarseSolveInfo(std::ostream& out) const
{
    assert(solver_[1]);

    if (myid_ == 0)
    {

        int num_solver = solver_.size();

        for (int i = 1; i < num_solver; ++i)
        {
            out << "\n";
            out << "Level " << i << " Solve Time:       " << solver_[i]->GetTiming() << "\n";
            out << "Level " << i << " Solve Iterations: " << solver_[i]->GetNumIterations() << "\n";
        }
    }
}

void GraphUpscale::ShowFineSolveInfo(std::ostream& out) const
{
    assert(solver_[0]);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Fine Solve Time:         " << solver_[0]->GetTiming() << "\n";
        out << "Fine Solve Iterations:   " << solver_[0]->GetNumIterations() << "\n";
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

double GraphUpscale::GetCoarseSolveTime() const
{
    assert(solver_[1]);

    return solver_[1]->GetTiming();
}

double GraphUpscale::GetFineSolveTime() const
{
    assert(solver_[0]);

    return solver_[0]->GetTiming();
}

int GraphUpscale::GetCoarseSolveIters() const
{
    assert(solver_[1]);

    return solver_[1]->GetNumIterations();
}

int GraphUpscale::GetFineSolveIters() const
{
    assert(solver_[0]);

    return solver_[0]->GetNumIterations();
}

double GraphUpscale::GetSetupTime() const
{
    return setup_time_;
}

std::vector<double> GraphUpscale::ComputeErrors(const BlockVector& upscaled_sol,
                                                const BlockVector& fine_sol) const
{
    const SparseMatrix& M = GetFineMatrix().LocalM();
    const SparseMatrix& D = GetFineMatrix().LocalD();

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

} // namespace smoothg
