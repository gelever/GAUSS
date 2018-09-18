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

/** @file UpscaleOperators.hpp
    @brief Useful wrappers for the upscale class

Each wrapper changes the operation performed by the Upscaler object.
This is useful when you need an operation other than just upscale.
Consider the mixed system:
\f[
\begin{bmatrix}
M & D^T \\
D & 0
\end{bmatrix}
\begin{bmatrix}
\sigma \\
u
\end{bmatrix}
=
\begin{bmatrix}
g  \\
-f
\end{bmatrix}
\f]

## GraphUpscale

The Upscaler usually solves for \f$ u \f$ by restricting to
the coarse level, solving, and interpolating back to the fine level.
The user only provides \f$ f \f$ and is returned \f$ u \f$.
The value of \f$ \sigma \f$ is discarded and \f$ g \f$ is always zero.

## Wrappers

### UpscaleBlockSolve
Solves for \f$ u \f$ and \f$ \sigma \f$ by restricting both to
the coarse level, solving, and interpolating both back to the fine level.
The user provides both \f$ g \f$ and \f$ f \f$ and is returned both \f$ \sigma \f$ and \f$ u \f$.

### UpscaleSolveLevel
Solves for \f$ u \f$ on the level by the provided solver.
The user provides \f$ f \f$  and is returned \f$ u \f$.

### UpscaleBlockSolveLevel
Solves for \f$ u \f$ and \f$ \sigma \f$ on the level by the provided solver.
The user provides both \f$ f \f$ and \f$ g \f$ and is returned \f$ u \f$ and \f$ \sigma \f$.

*/

#ifndef __UPSCALE_OPERATORS_HPP__
#define __UPSCALE_OPERATORS_HPP__

#include "Utilities.hpp"
#include "GraphUpscale.hpp"

namespace gauss
{

/// UpscaleBlockSolve performs the same thing as GraphUpscale, but in mixed form.
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleBlockSolve : public linalgcpp::Operator
{
public:
    UpscaleBlockSolve(const GraphUpscale& A) : linalgcpp::Operator(A.GetMatrix(0).Rows()), A_(A),
        x_(A_.BlockOffsets(0)), y_(A_.BlockOffsets(0)) { }

    void Mult(const VectorView& x, VectorView y) const
    {
        x_ = x;
        y_ = y;

        A_.Solve(x_, y_);

        y = y_;
    }

private:
    const GraphUpscale& A_;

    mutable BlockVector x_;
    mutable BlockVector y_;
};

/// UpscaleBlockSolveLevel Solves the problem on a particular level in the mixed form as its operation
/** @note All vectors assumed to be block vectors with the same offsets as the Upscaler */
class UpscaleBlockSolveLevel : public linalgcpp::Operator
{
public:
    UpscaleBlockSolveLevel(const GraphUpscale& A, int level)
        : linalgcpp::Operator(A.GetMatrix(level).Rows()),
          A_(A), level_(level)
    {
    }

    void Mult(const VectorView& x, VectorView y) const
    {
        x_ = x;
        y_ = y;

        A_.SolveLevel(level_, x_, y_);

        y = y_;
    }

private:
    const GraphUpscale& A_;

    mutable BlockVector x_;
    mutable BlockVector y_;

    int level_;
};

/// UpscaleSolveLevel Solves the problem on a particular level in the primal form as its operation
class UpscaleSolveLevel : public linalgcpp::Operator
{
public:
    UpscaleSolveLevel(const GraphUpscale& A, int level)
        : linalgcpp::Operator(A.GetMatrix(level).LocalD().Rows()),
          A_(A), level_(level) {}
    void Mult(const VectorView& x, VectorView y) const { A_.SolveLevel(level_, x, y); }

private:
    const GraphUpscale& A_;
    int level_;
};

} // namespace gauss

#endif // __UPSCALE_OPERATORS_HPP__
