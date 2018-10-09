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

#ifndef SPE10_HPP_
#define SPE10_HPP_

/**
   @file spe10.hpp
   @brief Implementation of spe10 problem.

   Reads data from file and creates the appropriate finite element structures.
*/

#include "mfem.hpp"
#include "GAUSS.hpp"
#include "Utilities.hpp"

namespace rs2001
{

/**
   @brief A utility class for working with the SPE10 data set.

   The SPE10 data set can be found at: http://www.spe.org/web/csp/datasets/set02.htm
*/
class InversePermeabilityFunction
{
public:

    enum SliceOrientation {NONE, XY, XZ, YZ};

    static void SetNumberCells(int Nx_, int Ny_, int Nz_);
    static void SetMeshSizes(double hx, double hy, double hz);
    static void Set2DSlice(SliceOrientation o, int npos );

    static void ReadPermeabilityFile(const std::string& fileName);
    static void ReadPermeabilityFile(const std::string& fileName, MPI_Comm comm);

    static void InversePermeability(const mfem::Vector& x, mfem::Vector& val);

    static double InvNorm2(const mfem::Vector& x);

    static void ClearMemory();

private:
    static int Nx;
    static int Ny;
    static int Nz;
    static double hx;
    static double hy;
    static double hz;
    static double* inversePermeability;

    static SliceOrientation orientation;
    static int npos;
};

/**
   @brief A forcing function that is supposed to very roughly represent some wells
   that are resolved on the *coarse* level.

   The forcing function is 1 on the top-left coarse cell, and -1 on the
   bottom-right coarse cell, and 0 elsewhere.

   @param Lx length of entire domain in x direction
   @param Hx size in x direction of a coarse cell.
*/
class GCoefficient : public mfem::Coefficient
{
public:
    GCoefficient(double Lx, double Ly, double Lz,
                 double Hx, double Hy, double Hz);
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip);
private:
    double Lx_, Ly_, Lz_;
    double Hx_, Hy_, Hz_;
};

/**
   @brief A function that marks half the resevior w/ value and the other -value.

   @param spe10_scale scale for length
*/
class HalfCoeffecient : public mfem::Coefficient
{
public:
    HalfCoeffecient(double value, int spe10_scale = 5)
        : value_(value), spe10_scale_(spe10_scale) {}
    double Eval(mfem::ElementTransformation& T,
                const mfem::IntegrationPoint& ip);
private:
    double value_;
    int spe10_scale_;
};

/**
   @brief Manages data from the SPE10 dataset.
*/
class SPE10Problem
{
public:
    SPE10Problem(const char* permFile, int nDimensions, int spe10_scale,
                 int slice, bool metis_partition, double proc_part_ubal,
                 const mfem::Array<double>& coarsening_factor);
    ~SPE10Problem();
    mfem::ParMesh* GetParMesh()
    {
        return pmesh_;
    }
    mfem::VectorFunctionCoefficient* GetKInv()
    {
        return kinv_;
    }
    GCoefficient* GetForceCoeff()
    {
        return source_coeff_;
    }
    const std::vector<int>& GetNumProcsXYZ()
    {
        return num_procs_xyz_;
    }
    static double CellVolume(int nDimensions)
    {
        return (nDimensions == 2 ) ? (20.0 * 10.0) : (20.0 * 10.0 * 2.0);
    }
private:
    double Lx, Ly, Lz, Hx, Hy, Hz;
    mfem::ParMesh* pmesh_;
    mfem::VectorFunctionCoefficient* kinv_;
    GCoefficient* source_coeff_;
    std::vector<int> num_procs_xyz_;
};

/**
   @brief Finite volume integrator

   This is the integrator for the artificial mass matrix in a finite
   volume discretization, tricking MFEM into doing finite volumes instead
   of finite elements.
*/
class FiniteVolumeMassIntegrator: public mfem::BilinearFormIntegrator
{
protected:
    mfem::Coefficient* Q;
    mfem::VectorCoefficient* VQ;
    mfem::MatrixCoefficient* MQ;

    // these are not thread-safe!
    mfem::Vector nor, ni;
    mfem::Vector unitnormal; // ATB 25 February 2015
    double sq;
    mfem::Vector vq;
    mfem::DenseMatrix mq;

public:
    ///@name Constructors differ by whether the coefficient (permeability) is scalar, vector, or full tensor
    ///@{
    FiniteVolumeMassIntegrator() :
        Q(NULL), VQ(NULL), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::Coefficient& q) :
        Q(&q), VQ(NULL), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::VectorCoefficient& q) :
        Q(NULL), VQ(&q), MQ(NULL)
    {
    }
    FiniteVolumeMassIntegrator(mfem::MatrixCoefficient& q) :
        Q(NULL), VQ(NULL), MQ(&q)
    {
    }
    ///@}

    using mfem::BilinearFormIntegrator::AssembleElementMatrix;
    /// Implements interface for MFEM's BilinearForm
    virtual void AssembleElementMatrix (const mfem::FiniteElement& el,
                                        mfem::ElementTransformation& Trans,
                                        mfem::DenseMatrix& elmat);
}; // class FiniteVolumeMassIntegrator

} // namespace rs2001

#endif /* SPE10_HPP_ */

