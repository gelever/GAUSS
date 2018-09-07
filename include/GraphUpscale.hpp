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

#ifndef __GRAPHUPSCALE_HPP__
#define __GRAPHUPSCALE_HPP__

#include <utility>

#include "linalgcpp.hpp"
#include "parlinalgcpp.hpp"
#include "partition.hpp"


#include "Utilities.hpp"
#include "MixedMatrix.hpp"
#include "GraphCoarsen.hpp"
#include "MGLSolver.hpp"
#include "Graph.hpp"
#include "Level.hpp"

#include "MinresBlockSolver.hpp"
#include "HybridSolver.hpp"
#include "SPDSolver.hpp"

namespace smoothg
{

/**
   @brief Collection of parameters for GraphUpscale
*/
struct UpscaleParams
{
    /**
       @brief Default Constructor
    */
    UpscaleParams() : UpscaleParams(0.001, 4) { }

    /**
       @brief Individual Parameter Constructor

       @param spect_tol_in spectral tolerance determines how many eigenvectors to
                        keep per aggregate
       @param max_evects_in maximum number of eigenvectors to keep per aggregate
       @param hybridization_in use hybridization as solver
       @param max_levels_in maximum number of levels to coarsen
       @param coarsen_factor_in metis coarsening factor if using multilevel upscaling
       @param elim_edge_dofs_in edge dofs to eliminate on the fine level
    */
    UpscaleParams(double spect_tol_in, int max_evects_in, bool hybridization_in = false,
                  int max_levels_in = 2, double coarsen_factor_in = 4.0,
                  const std::vector<int>& elim_edge_dofs_in = {})
        : hybridization(hybridization_in),
          max_levels(max_levels_in), coarsen_factor(coarsen_factor_in),
          spectral_pair(max_levels_in - 1, {spect_tol_in, max_evects_in}),
          elim_edge_dofs(elim_edge_dofs_in)
          { }

    bool hybridization;
    int max_levels;
    double coarsen_factor;

    std::vector<SpectralPair> spectral_pair;
    std::vector<int> elim_edge_dofs;
};


/**
   @brief Use upscaling as operator
*/
class GraphUpscale : public linalgcpp::Operator
{
public:
    /// Default Constructor
    GraphUpscale() = default;

    /**
       @brief Graph Constructor

       @param graph input graph information
       @param params set of upscaling parameters
    */
    GraphUpscale(const Graph& graph, const UpscaleParams& params = {});

    /// Default Destructor
    ~GraphUpscale() = default;

    /// Get global number of rows (vertex dofs)
    int GlobalRows() const;

    /// Get global number of columns (vertex dofs)
    int GlobalCols() const;

    /// Create Solver
    void MakeSolver(int level);
    void MakeSolver(int level, MixedMatrix& mm);

    /// Rescale solver w/ weights per aggregate
    void RescaleSolver(int level, const std::vector<double>& agg_weights);
    void RescaleSolver(int level, const std::vector<double>& agg_weights,
                    MixedMatrix& mm);

    /// Wrapper for applying the upscaling, in linalgcpp terminology
    void Mult(const VectorView& x, VectorView y) const override;

    void MultMultiLevel(const BlockVector& x, std::vector<BlockVector>& sols) const;
    std::vector<BlockVector> MultMultiLevel(const BlockVector& x) const;

    void MultMultiGrid(const BlockVector& x, BlockVector& y) const;
    BlockVector MultMultiGrid(const BlockVector& x) const;

    /// Wrapper for applying the upscaling
    void Solve(const VectorView& x, VectorView y) const;
    Vector Solve(const VectorView& x) const;

    /// Wrapper for applying the upscaling in mixed form
    void Solve(const BlockVector& x, BlockVector& y) const;
    BlockVector Solve(const BlockVector& x) const;

    /// Upscaled Solution from Level
    void Solve(int level, const VectorView& x, VectorView y) const;
    Vector Solve(int level, const VectorView& x) const;

    /// Upscaled Solution from Level, in mixed form
    void Solve(int level, const BlockVector& x, BlockVector& y) const;
    BlockVector Solve(int level, const BlockVector& x) const;

    /// Solve Level only, no restriction/interpolate
    void SolveLevel(int level, const VectorView& x, VectorView y) const;
    Vector SolveLevel(int level, const VectorView& x) const;

    /// Solve Level only, in mixed form
    void SolveLevel(int level, const BlockVector& x, BlockVector& y) const;
    BlockVector SolveLevel(int level, const BlockVector& x) const;

    /// Interpolate a coarse vector to the fine level
    void Interpolate(const VectorView& x, VectorView y) const;
    Vector Interpolate(const VectorView& x, int level = 0) const;

    /// Interpolate a coarse vector to the fine level, in mixed form
    void Interpolate(const BlockVector& x, BlockVector& y) const;
    BlockVector Interpolate(const BlockVector& x, int level = 0) const;

    /// Restrict a fine vector to the coarse level
    void Restrict(const VectorView& x, VectorView y) const;
    Vector Restrict(const VectorView& x, int level = 1) const;

    /// Restrict a fine vector to the coarse level, in mixed form
    void Restrict(const BlockVector& x, BlockVector& y) const;
    BlockVector Restrict(const BlockVector& x, int level = 1) const;

    /// Project a fine vector to the coarse level, in mixed form
    void Project(const BlockVector& x, BlockVector& y) const;
    BlockVector Project(const BlockVector& x) const;

    /// Get block offsets
    const std::vector<int>& BlockOffsets(int level) const;
    const std::vector<int>& TrueBlockOffsets(int level) const;

    /// Orthogonalize against the constant vector
    void Orthogonalize(int level, VectorView vect) const;
    void Orthogonalize(int level, BlockVector& vect) const;

    /// Create vectors by level
    Vector GetVector(int level) const;
    BlockVector GetBlockVector(int level) const;
    BlockVector GetTrueBlockVector(int level) const;

    std::vector<Vector> GetMLVectors() const;
    std::vector<BlockVector> GetMLBlockVector() const;

    /// Get Matrix by level
    MixedMatrix& GetMatrix(int level);
    const MixedMatrix& GetMatrix(int level) const;

    /// Get GraphSpace by level
    GraphSpace& GetGraphSpace(int level) { return GetLevel(level).graph_space; }
    const GraphSpace& GetGraphSpace(int level) const { return GetLevel(level).graph_space; }

    /// Show Solver Information
    void PrintInfo(std::ostream& out = std::cout) const;

    /// Compute Operator Complexity
    double OperatorComplexity() const;

    /// Get communicator
    MPI_Comm GetComm() const { return comm_; }

    /// Set solver parameters
    void SetPrintLevel(int print_level);
    void SetMaxIter(int max_num_iter);
    void SetRelTol(double rtol);
    void SetAbsTol(double atol);

    /// Show Total Solve time on the coarse level on processor 0
    void ShowCoarseSolveInfo(std::ostream& out = std::cout) const;

    /// Show Total Solve time on the fine level on processor 0
    void ShowFineSolveInfo(std::ostream& out = std::cout) const;

    /// Show Total setup time on processor 0
    void ShowSetupTime(std::ostream& out = std::cout) const;

    /// Get Solve time on the level for the last solve
    double SolveTime(int level) const;

    /// Get Solve iterations on the level for the last solve
    int SolveIters(int level) const;

    /// Get Total setup time
    double GetSetupTime() const;

    /// Get Coarsener
    const GraphCoarsen& Coarsener(int level) const { return coarsener_.at(level); }

    /// Get Solver
    const MGLSolver& Solver(int level) const { return *GetLevel(level).solver; }

    /// Get Level
    Level& GetLevel(int level) { return levels_.at(level); }
    const Level& GetLevel(int level) const { return levels_.at(level); }

    /// Number of levels
    int NumLevels() const { return levels_.size(); }

    /// Get Normalized Constant Representation
    const Vector& ConstantRep(int level) const { return GetLevel(level).constant_rep; }

    /// Compare errors between upscaled and fine solution.
    /// Returns {vertex_error, edge_error, div_error} array.
    std::vector<double> ComputeErrors(const BlockVector& upscaled_sol,
                                      const BlockVector& fine_sol) const;

    /// Compare errors between upscaled and fine solution.
    /// Displays error to stdout on processor 0
    void ShowErrors(const BlockVector& upscaled_sol,
                    const BlockVector& fine_sol) const;

    ParMatrix ToPrimal() const;

    /// Set orthogonalization against constant of vertex solution after solving
    void SetOrthogonalize(bool do_ortho = true) { do_ortho_ = do_ortho; }

    /// Check use of hybridizaiton solver
    bool Hybridization() const { return hybridization_; }

    /// Check use of orthogonalization
    bool Orthogonalization() const { return do_ortho_; }

private:
    std::vector<Level> levels_;
    std::vector<GraphCoarsen> coarsener_;

    MPI_Comm comm_;
    int myid_;

    double setup_time_;

    std::unordered_map<int, int> size_to_level_;

    bool hybridization_;
    bool do_ortho_;
};

} // namespace smoothg

#endif /* __GRAPHUPSCALE_HPP__ */
