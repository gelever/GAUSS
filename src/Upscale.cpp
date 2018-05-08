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

    @brief Contains Upscale class
*/

#include "Upscale.hpp"

namespace smoothg
{

Upscale::Upscale(const Graph& graph)
    : comm_(graph.edge_true_edge_.GetComm()),
      global_vertices_(graph.global_vertices_),
      global_edges_(graph.global_edges_),
      setup_time_(0)
{
    MPI_Comm_size(comm_, &num_procs_);
    MPI_Comm_rank(comm_, &myid_);
}

void Upscale::Mult(const VectorView& x, VectorView y) const
{
    assert(coarse_solver_);

    coarsener_.Restrict(x, rhs_coarse_.GetBlock(1));

    rhs_coarse_.GetBlock(0) = 0.0;
    rhs_coarse_.GetBlock(1) *= -1.0;

    coarse_solver_->Solve(rhs_coarse_, sol_coarse_);

    coarsener_.Interpolate(sol_coarse_.GetBlock(1), y);

    Orthogonalize(y);
}

void Upscale::Solve(const VectorView& x, VectorView y) const
{
    Mult(x, y);
}

Vector Upscale::Solve(const VectorView& x) const
{
    Vector y(x.size());

    Solve(x, y);

    return y;
}

void Upscale::Solve(const BlockVector& x, BlockVector& y) const
{
    assert(coarse_solver_);

    coarsener_.Restrict(x, rhs_coarse_);
    rhs_coarse_.GetBlock(1) *= -1.0;

    coarse_solver_->Solve(rhs_coarse_, sol_coarse_);
    coarsener_.Interpolate(sol_coarse_, y);

    Orthogonalize(y);
}

BlockVector Upscale::Solve(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    Solve(x, y);

    return y;
}

void Upscale::SolveCoarse(const VectorView& x, VectorView y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
}

Vector Upscale::SolveCoarse(const VectorView& x) const
{
    Vector coarse_vect = GetCoarseVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveCoarse(const BlockVector& x, BlockVector& y) const
{
    assert(coarse_solver_);

    coarse_solver_->Solve(x, y);
    y *= -1.0;
}

BlockVector Upscale::SolveCoarse(const BlockVector& x) const
{
    BlockVector coarse_vect = GetCoarseBlockVector();
    SolveCoarse(x, coarse_vect);

    return coarse_vect;
}

void Upscale::SolveFine(const VectorView& x, VectorView y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    y *= -1.0;

    Orthogonalize(y);
}

Vector Upscale::SolveFine(const VectorView& x) const
{
    Vector y(x.size());

    SolveFine(x, y);

    return y;
}

void Upscale::SolveFine(const BlockVector& x, BlockVector& y) const
{
    assert(fine_solver_);

    fine_solver_->Solve(x, y);
    y *= -1.0;

    Orthogonalize(y);
}

BlockVector Upscale::SolveFine(const BlockVector& x) const
{
    BlockVector y = GetFineBlockVector();

    SolveFine(x, y);

    return y;
}

void Upscale::Interpolate(const VectorView& x, VectorView y) const
{
    coarsener_.Interpolate(x, y);
}

Vector Upscale::Interpolate(const VectorView& x) const
{
    return coarsener_.Interpolate(x);
}

void Upscale::Interpolate(const BlockVector& x, BlockVector& y) const
{
    coarsener_.Interpolate(x, y);
}

BlockVector Upscale::Interpolate(const BlockVector& x) const
{
    return coarsener_.Interpolate(x);
}

void Upscale::Restrict(const VectorView& x, VectorView y) const
{
    coarsener_.Restrict(x, y);
}

Vector Upscale::Restrict(const VectorView& x) const
{
    return coarsener_.Restrict(x);
}

void Upscale::Restrict(const BlockVector& x, BlockVector& y) const
{
    coarsener_.Restrict(x, y);
}

BlockVector Upscale::Restrict(const BlockVector& x) const
{
    return coarsener_.Restrict(x);
}

const std::vector<int>& Upscale::FineBlockOffsets() const
{
    return GetFineMatrix().Offsets();
}

const std::vector<int>& Upscale::CoarseBlockOffsets() const
{
    return GetCoarseMatrix().Offsets();
}

const std::vector<int>& Upscale::FineTrueBlockOffsets() const
{
    return GetFineMatrix().TrueOffsets();
}

const std::vector<int>& Upscale::CoarseTrueBlockOffsets() const
{
    return GetCoarseMatrix().TrueOffsets();
}

void Upscale::Orthogonalize(VectorView vect) const
{
    OrthoConstant(comm_, vect, GetFineMatrix().GlobalD().GlobalRows());
}

void Upscale::Orthogonalize(BlockVector& vect) const
{
    Orthogonalize(vect.GetBlock(1));
}

Vector Upscale::GetCoarseVector() const
{
    int coarse_size = GetCoarseMatrix().LocalD().Rows();

    return Vector(coarse_size);
}

Vector Upscale::GetFineVector() const
{
    int fine_size = GetFineMatrix().LocalD().Rows();

    return Vector(fine_size);
}

BlockVector Upscale::GetCoarseBlockVector() const
{
    return BlockVector(GetCoarseMatrix().Offsets());
}

BlockVector Upscale::GetFineBlockVector() const
{
    return BlockVector(GetFineMatrix().Offsets());
}

BlockVector Upscale::GetCoarseTrueBlockVector() const
{
    return BlockVector(GetCoarseMatrix().TrueOffsets());
}

BlockVector Upscale::GetFineTrueBlockVector() const
{
    return BlockVector(GetFineMatrix().TrueOffsets());
}

MixedMatrix& Upscale::GetMatrix(int level)
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));
    assert(mgl_[level]);

    return *mgl_[level];
}

const MixedMatrix& Upscale::GetMatrix(int level) const
{
    assert(level >= 0 && level < static_cast<int>(mgl_.size()));
    assert(mgl_[level]);

    return *mgl_[level];
}

MixedMatrix& Upscale::GetFineMatrix()
{
    return GetMatrix(0);
}

const MixedMatrix& Upscale::GetFineMatrix() const
{
    return GetMatrix(0);
}

MixedMatrix& Upscale::GetCoarseMatrix()
{
    return GetMatrix(1);
}

const MixedMatrix& Upscale::GetCoarseMatrix() const
{
    return GetMatrix(1);
}

void Upscale::PrintInfo(std::ostream& out) const
{
    // Matrix sizes, not solvers
    int nnz_coarse = GetCoarseMatrix().GlobalNNZ();
    int nnz_fine = GetFineMatrix().GlobalNNZ();

    // True dof size
    int size_fine = GetFineMatrix().GlobalRows();
    int size_coarse = GetCoarseMatrix().GlobalRows();

    int num_procs;
    MPI_Comm_size(comm_, &num_procs);

    double op_comp = OperatorComplexity();

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

        out << "Fine Matrix\n";
        out << "---------------------\n";
        out << "Size\t\t" << size_fine << "\n";
        out << "NonZeros:\t" << nnz_fine << "\n";
        out << "\n";
        out << "Coarse Matrix\n";
        out << "---------------------\n";
        out << "Size\t\t" << size_coarse << "\n";
        out << "NonZeros:\t" << nnz_coarse << "\n";
        out << "\n";
        out << "Op Comp:\t" << op_comp << "\n";

        out.precision(old_precision);
    }
}

double Upscale::OperatorComplexity() const
{
    assert(coarse_solver_);

    int nnz_coarse = coarse_solver_->GetNNZ();
    int nnz_fine;

    if (fine_solver_)
    {
        nnz_fine = fine_solver_->GetNNZ();
    }
    else
    {
        nnz_fine = GetFineMatrix().GlobalNNZ();
    }


    double op_comp = 1.0 + (nnz_coarse / (double) nnz_fine);

    return op_comp;
}

void Upscale::SetPrintLevel(int print_level)
{
    assert(coarse_solver_);
    coarse_solver_->SetPrintLevel(print_level);

    if (fine_solver_)
    {
        fine_solver_->SetPrintLevel(print_level);
    }
}

void Upscale::SetMaxIter(int max_num_iter)
{
    assert(coarse_solver_);
    coarse_solver_->SetMaxIter(max_num_iter);

    if (fine_solver_)
    {
        fine_solver_->SetMaxIter(max_num_iter);
    }
}

void Upscale::SetRelTol(double rtol)
{
    assert(coarse_solver_);
    coarse_solver_->SetRelTol(rtol);

    if (fine_solver_)
    {
        fine_solver_->SetRelTol(rtol);
    }
}

void Upscale::SetAbsTol(double atol)
{
    assert(coarse_solver_);
    coarse_solver_->SetAbsTol(atol);

    if (fine_solver_)
    {
        fine_solver_->SetAbsTol(atol);
    }
}

void Upscale::ShowCoarseSolveInfo(std::ostream& out) const
{
    assert(coarse_solver_);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Coarse Solve Time:       " << coarse_solver_->GetTiming() << "\n";
        out << "Coarse Solve Iterations: " << coarse_solver_->GetNumIterations() << "\n";
    }
}

void Upscale::ShowFineSolveInfo(std::ostream& out) const
{
    assert(fine_solver_);

    if (myid_ == 0)
    {
        out << "\n";
        out << "Fine Solve Time:         " << fine_solver_->GetTiming() << "\n";
        out << "Fine Solve Iterations:   " << fine_solver_->GetNumIterations() << "\n";
    }
}

void Upscale::ShowSetupTime(std::ostream& out) const
{
    if (myid_ == 0)
    {
        out << "\n";
        out << "Upscale Setup Time:      " << setup_time_ << "\n";
    }
}

double Upscale::GetCoarseSolveTime() const
{
    assert(coarse_solver_);

    return coarse_solver_->GetTiming();
}

double Upscale::GetFineSolveTime() const
{
    assert(fine_solver_);

    return fine_solver_->GetTiming();
}

int Upscale::GetCoarseSolveIters() const
{
    assert(coarse_solver_);

    return coarse_solver_->GetNumIterations();
}

int Upscale::GetFineSolveIters() const
{
    assert(fine_solver_);

    return fine_solver_->GetNumIterations();
}

double Upscale::GetSetupTime() const
{
    return setup_time_;
}

std::vector<double> Upscale::ComputeErrors(const BlockVector& upscaled_sol,
                                           const BlockVector& fine_sol) const
{
    const SparseMatrix& M = GetFineMatrix().LocalM();
    const SparseMatrix& D = GetFineMatrix().LocalD();

    auto info = smoothg::ComputeErrors(comm_, M, D, upscaled_sol, fine_sol);
    info.push_back(OperatorComplexity());

    return info;
}

void Upscale::ShowErrors(const BlockVector& upscaled_sol,
                         const BlockVector& fine_sol) const
{
    auto info = ComputeErrors(upscaled_sol, fine_sol);

    if (myid_ == 0)
    {
        smoothg::ShowErrors(info);
    }
}

void Upscale::MakeCoarseVectors()
{
    rhs_coarse_ = BlockVector(GetCoarseMatrix().Offsets());
    sol_coarse_ = BlockVector(GetCoarseMatrix().Offsets());
}

} // namespace smoothg
