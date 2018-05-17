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
   @file Sampler.hpp
   @brief Contains sampler implementations
*/

#ifndef __SAMPLER_HPP__
#define __SAMPLER_HPP__

#include "smoothG.hpp"

namespace smoothg
{

class NormalDistribution
{
    public:
        NormalDistribution(double mean = 0.0, double stddev = 1.0, int seed = 0)
            : generator_(seed), dist_(mean, stddev) { }

        ~NormalDistribution() = default;

        double Sample() { return dist_(generator_); }

    private:
        std::mt19937 generator_;
        std::normal_distribution<double> dist_;
};

class SamplerUpscale
{
    public:

        SamplerUpscale(Graph graph, double spect_tol, int max_evects, bool hybridization,
                       int dimension, double kappa, double cell_volume, int seed);

        ~SamplerUpscale() = default;

        void Sample();

        const std::vector<double>& GetCoefficientFine() const { return coefficient_fine_; }
        const std::vector<double>& GetCoefficientCoarse() const { return coefficient_coarse_; }
        const std::vector<double>& GetCoefficientUpscaled() const { return coefficient_upscaled_; }

        const GraphUpscale& GetUpscale() const { return upscale_; }

    private:
        GraphUpscale upscale_;

        NormalDistribution normal_dist_;
        double cell_volume_;
        double scalar_g_;

        Vector rhs_fine_;
        Vector rhs_coarse_;

        Vector sol_fine_;
        Vector sol_coarse_;

        std::vector<double> coefficient_fine_;
        std::vector<double> coefficient_coarse_;
        std::vector<double> coefficient_upscaled_;

        Vector constant_coarse_;
};


SamplerUpscale::SamplerUpscale(Graph graph, double spect_tol, int max_evects, bool hybridization,
                               int dimension, double kappa, double cell_volume, int seed)
    : upscale_(std::move(graph), spect_tol, max_evects, hybridization),
      normal_dist_(0.0, 1.0, seed),
      cell_volume_(cell_volume),
      rhs_fine_(upscale_.GetFineVector()),
      rhs_coarse_(upscale_.GetCoarseVector()),
      sol_fine_(upscale_.GetFineVector()),
      sol_coarse_(upscale_.GetCoarseVector()),
      coefficient_fine_(upscale_.Rows()),
      coefficient_coarse_(upscale_.NumAggs()),
      coefficient_upscaled_(upscale_.Rows()),
      constant_coarse_(upscale_.GetCoarseVector())
{
    upscale_.PrintInfo();
    upscale_.ShowSetupTime();

    Vector ones = upscale_.GetFineVector();
    ones = 1.0;

    upscale_.Restrict(ones, constant_coarse_);

    double nu_param = dimension == 2 ? 1.0 : 0.5;
    double ddim = static_cast<double>(dimension);

    scalar_g_ = std::pow(4.0 * M_PI, ddim / 4.0) * std::pow(kappa, nu_param) *
            std::sqrt( std::tgamma(nu_param + ddim / 2.0) / std::tgamma(nu_param) );
}

void SamplerUpscale::Sample()
{
    double g_cell_vol_sqrt = scalar_g_ * std::sqrt(cell_volume_);

    for (auto& i : rhs_fine_)
    {
        i = g_cell_vol_sqrt * normal_dist_.Sample();
    }

    // Set Fine Coefficient
    upscale_.SolveFine(rhs_fine_, sol_fine_);

    int fine_size = sol_fine_.size();
    assert(coefficient_fine_.size() == fine_size);

    for (int i = 0; i < fine_size; ++i)
    {
        coefficient_fine_[i] = std::exp(sol_fine_[i]);
    }

    // Set Coarse Coefficient
    upscale_.Restrict(rhs_fine_, rhs_coarse_);
    upscale_.SolveCoarse(rhs_coarse_, sol_coarse_);

    int coarse_size = sol_coarse_.size();
    assert(constant_coarse_.size() == coarse_size);

    std::fill(std::begin(coefficient_coarse_), std::end(coefficient_coarse_), 0.0);
    int agg_index = 0;

    for (int i = 0; i < coarse_size; ++i)
    {
        if (std::fabs(constant_coarse_[i]) > 1e-8)
        {
            sol_coarse_[i] = std::exp(sol_coarse_[i] / constant_coarse_[i]);
            coefficient_coarse_[agg_index++] = sol_coarse_[i];
        }
        else
        {
            sol_coarse_[i] = 0.0;
        }
    }

    assert(agg_index == upscale_.NumAggs());

    // Set Upscaled Coefficient
    sol_coarse_ *= constant_coarse_;
    VectorView coeff_view(coefficient_upscaled_.data(), coefficient_upscaled_.size());
    upscale_.Interpolate(sol_coarse_, coeff_view);
}


} // namespace smoothg

#endif // __SAMPLER_HPP__
