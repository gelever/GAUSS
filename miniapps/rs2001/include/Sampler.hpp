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

/**
   @file Sampler.hpp
   @brief Contains sampler implementations
*/

#ifndef __SAMPLER_HPP__
#define __SAMPLER_HPP__

#include <unordered_map>

#include "spe10.hpp"

namespace rs2001
{

/** @brief Saves output vectors to file as ("prefix" + index + ".txt")
    @param upscale upscale object to perform permutations
    @param vect local vector to save
    @param prefix filename prefix
    @param index filename suffix
*/
template <typename T>
void SaveOutput(const gauss::Graph& graph, const T& vect, const std::string& prefix, int index)
{
    std::stringstream ss;
    ss << prefix << std::setw(5) << std::setfill('0') << index << ".txt";

    WriteVertexVector(graph, vect, ss.str());
}

/** @brief Scalar normal distribution */
class NormalDistribution
{
public:
    /** @brief Constructor setting RNG paramaters
        @param mean mean
        @param stddev standard deviation
        @param seed generator seed
    */
    NormalDistribution(double mean = 0.0, double stddev = 1.0, int seed = 0)
        : generator_((seed >= 0) ? seed : (std::random_device())()), dist_(mean, stddev) { }
    /* https://kristerw.blogspot.com/2017/05/seeding-stdmt19937-random-number-engine.html
        : dist_(mean, stddev)
    {
        std::random_device rd;
        std::array<int, std::mt19937::state_size> seed_data;
        std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
        generator_ = std::mt19937(seq);
    }
    */

    /** @brief Default Destructor */
    ~NormalDistribution() = default;

    /** @brief Generate a random number */
    double Sample() { return dist_(generator_); }
    void operator()(mfem::Vector& xi)
    {
        int size = xi.Size();
        for (int i = 0; i < size; ++i)
        {
            xi[i] = Sample();
        }
    }

private:
    std::mt19937 generator_;
    std::normal_distribution<double> dist_;
};


/** @brief Provides lognormal random fields with Matern covariance.

    Uses technique from Osborn, Vassilevski, and Villa, A multilevel,
    hierarchical sampling technique for spatially correlated random fields,
    SISC 39 (2017) pp. S543-S562.
*/
class PDESampler
{
public:

    /** @brief Constructor w/ given graph information and upscaling params
        @param graph Graph information
        @param double spect_tol spectral tolerance for upscaling
        @param max_evects maximum number of eigenvectors for upscaling
        @param hybridization use hybridization solver
        @param dimension spatial dimension of the mesh from which the graph originates
        @param cell_volume size of a typical cell
        @param kappa inverse correlation length for Matern covariance
        @param seed seed for random number generator
     */
    PDESampler(const gauss::Graph& graph, const gauss::UpscaleParams& params,
               int dimension, double kappa, double cell_volume, bool lognormal = true);

    /** @brief Default Destructor */
    ~PDESampler() = default;

    /** @brief Generate a new sample.
        @param coarse_sample generate the sample on the coarse level
    */
    void Sample(bool coarse_sample = false);

    /** @brief Access the coefficients */
    const std::vector<double>& GetCoefficient(int level) const { return coefficient_.at(level); }

    const std::vector<gauss::Vector>& GetUpscaledCoefficient() const { return upscaled_coeff_; }


    /** @brief Access the GraphUpscale object */
    const gauss::GraphUpscale& GetUpscale() const { return upscale_; }

    /** @brief Get the total number of solver iterations. */
    int TotalIters(int level) const { return solve_iters_.at(level); }

    /** @brief Get the total solve time of the solver. */
    double TotalTime(int level) const { return solve_time_.at(level); }

    /// Fill vector with random sample using dist_sampler
    void Sample(const int level, mfem::Vector& xi);

    /// Get solve time for last sample
    double SampleTime(int level) const { return upscale_.SolveTime(level); }

    /// Evaluate random field at level with random sample xi
    virtual void Eval(
        const int level,
        const mfem::Vector& xi,
        mfem::Vector& s);

    virtual void Eval(
        const int level,
        const mfem::Vector& xi,
        mfem::Vector& s,
        mfem::Vector& u,
        bool use_init);

    virtual int SampleSize(int level) const { return upscale_.GetMatrix(level).GetElemDof().Rows(); }

    virtual size_t GetNNZ(int level) const { return upscale_.Solver(level).GetNNZ(); }

    /// Compute L2 Error of coeff and exact soln (double) for level
    virtual double ComputeL2Error(
        int level,
        const mfem::Vector& coeff,
        double exact) const;

    /** @brief Access the FESpace object */
    const mfem::FiniteElementSpace* FESpace() const { return fespace_; }
    void SetFESpace(mfem::FiniteElementSpace* fespace) { fespace_ = fespace; }

    int GlobalSampleSize(int level) { return global_sample_size_[level]; }
    int GetNumberOfDofs(int level) { return upscale_.GetMatrix(level).Rows(); }
    int GetGlobalNumberOfDofs(int level) { return upscale_.GetMatrix(level).GlobalRows(); }

    gauss::Vector Interpolate(int level, const mfem::Vector& coeff) const;
    mfem::Vector Restrict(int level, const mfem::Vector& coeff) const;


private:
    gauss::GraphUpscale upscale_;

    NormalDistribution normal_dist_;
    double cell_volume_;

    double alpha_;
    double matern_coeff_;
    bool log_normal_;

    std::vector<std::vector<int>> constant_map_;

    std::vector<gauss::Vector> rhs_;
    std::vector<gauss::Vector> sol_;

    std::vector<gauss::Vector> constant_rep_;
    std::unordered_map<int, int> size_to_level_;

    std::vector<std::vector<double>> Ws_;

    std::vector<std::vector<double>> coefficient_;
    std::vector<gauss::Vector> upscaled_coeff_;

    std::vector<int> solve_iters_;
    std::vector<double> solve_time_;

    mfem::FiniteElementSpace* fespace_;

    std::vector<int> global_sample_size_;
};

} // namespace rs2001

#endif // __SAMPLER_HPP__
