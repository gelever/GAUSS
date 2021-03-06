/*BHEADER**********************************************************************
 *
 * Copyright (c) 2018, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-759464. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of GAUSS. For more information and source code
 * availability, see https://www.github.com/gelever/GAUSS.
 *
 * GAUSS is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 ***********************************************************************EHEADER*/

/** @file

    @brief Contains abstract base class MGLSolver,
           Mixed Graph Laplacian Solver
*/

#ifndef __MGLSOLVER_HPP__
#define __MGLSOLVER_HPP__

#include "Utilities.hpp"
#include "MixedMatrix.hpp"

namespace gauss
{

/**
   @brief Abstract base class for solvers of graph Laplacian problems
*/
class MGLSolver : public linalgcpp::Operator
{
public:
    /** @brief Default Constructor */
    MGLSolver() = default;

    /** @brief Constructor from a mixed matrix
        @param mgl mixed matrix information
    */
    MGLSolver(const MixedMatrix& mgl);

    /** @brief Copy Constructor */
    MGLSolver(const MGLSolver& other) noexcept;

    /** @brief Swap two solvers */
    friend void swap(MGLSolver& lhs, MGLSolver& rhs) noexcept;

    /** @brief Default Destructor */
    virtual ~MGLSolver() = default;

    /**
       @brief Solve the graph Laplacian problem in mixed form

       The BlockVectors here are in "dof" numbering, rather than "truedof" numbering.
       That is, dofs on processor boundaries are *repeated* in the vectors that
       come into and go out of this method.

       @param rhs Right hand side
       @param sol Solution
    */
    virtual void Solve(const BlockVector& rhs, BlockVector& sol) const = 0;
    virtual void Mult(const BlockVector& rhs, BlockVector& sol) const;
    virtual BlockVector Mult(const BlockVector& rhs) const;

    /**
       @brief Solve the graph Laplacian problem as primal.
              Only vertex data is used, edge vectors are not considered.

       @param rhs Right hand side
       @param sol Solution
    */
    virtual void Solve(const VectorView& rhs, VectorView sol) const;
    virtual void Mult(const VectorView& rhs, VectorView sol) const;
    virtual Vector Mult(const VectorView& rhs) const;

    using linalgcpp::Operator::Mult;

    ///@name Set solver parameters
    ///@{
    virtual void SetPrintLevel(int print_level) { print_level_ = print_level; }
    virtual void SetMaxIter(int max_num_iter) { max_num_iter_ = max_num_iter; }
    virtual void SetRelTol(double rtol) { rtol_ = rtol; }
    virtual void SetAbsTol(double atol) { atol_ = atol; }
    ///@}

    ///@name Get results of iterative solve
    ///@{
    virtual int GetNumIterations() const { return num_iterations_; }
    virtual int GetNNZ() const { return nnz_; }
    virtual double GetTiming() const { return timing_; }
    ///@}

protected:
    MPI_Comm comm_;
    int myid_;
    bool use_w_;

    mutable BlockVector rhs_;
    mutable BlockVector sol_;

    // default linear solver options
    int print_level_ = 0;
    int max_num_iter_ = 5000;
    double rtol_ = 1e-9;
    double atol_ = 1e-12;

    int nnz_ = 0;
    mutable int num_iterations_ = 0;
    mutable double timing_ = 0;
};

} // namespace gauss

#endif // __MGLSOLVER_HPP__
