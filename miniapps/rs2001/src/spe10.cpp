#include "spe10.hpp"

namespace rs2001
{

GCoefficient::GCoefficient(double Lx, double Ly, double Lz,
                           double Hx, double Hy, double Hz)
    :
    Lx_(Lx),
    Ly_(Ly),
    Lz_(Lz),
    Hx_(Hx),
    Hy_(Hy),
    Hz_(Hz)
{
}

double GCoefficient::Eval(mfem::ElementTransformation& T,
                          const mfem::IntegrationPoint& ip)
{
    double dx[3];
    mfem::Vector transip(dx, 3);

    T.Transform(ip, transip);

    if ((transip(0) < Hx_) && (transip(1) > (Ly_ - Hy_)))
        return 1.0;
    else if ((transip(0) > (Lx_ - Hx_)) && (transip(1) < Hy_))
        return -1.0;
    return 0.0;
}

double HalfCoeffecient::Eval(mfem::ElementTransformation& T,
                             const mfem::IntegrationPoint& ip)
{
    double dx[3];
    mfem::Vector transip(dx, 3);

    T.Transform(ip, transip);

    //return transip(1) < (spe10_scale_ * 44 * 5) ? -value_ : value_;
    return (transip(1) < (spe10_scale_ * 22 * 5) || transip(1) > (spe10_scale_ * 66 * 5)
            || transip(0) < (spe10_scale_ * 60) || transip(0) > (spe10_scale_ * 180)) ? -value_ : value_;
}



SPE10Problem::SPE10Problem(const char* permFile, int nDimensions,
                           int spe10_scale, int slice,  bool metis_partition, double proc_part_ubal,
                           const mfem::Array<double>& coarsening_factor)
{
    int num_procs, myid;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::Array<int> N(3);
    N[0] = 12 * spe10_scale; // 60
    N[1] = 44 * spe10_scale; // 220
    N[2] = 17 * spe10_scale; // 85

    // SPE10 grid cell dimensions
    mfem::Vector h(3);
    h(0) = 20.0;
    h(1) = 10.0;
    h(2) = 2.0;
    unique_ptr<mfem::Mesh> mesh;

    using IPF = InversePermeabilityFunction;

    IPF::SetNumberCells(N[0], N[1], N[2]);
    IPF::SetMeshSizes(h(0), h(1), h(2));
    IPF::ReadPermeabilityFile(permFile, MPI_COMM_WORLD);

    if (nDimensions == 2)
        IPF::Set2DSlice(IPF::XY, slice);

    kinv_ = new mfem::VectorFunctionCoefficient(
        nDimensions, IPF::InversePermeability);

    const bool use_egg_model = false;
    if (use_egg_model)
    {
        std::string meshfile = "Egg_model.mesh";
        std::ifstream imesh(meshfile.c_str());
        if (!imesh)
        {
            if (myid == 0)
                std::cerr << "\nCan not open mesh file: " << meshfile
                          << std::endl;
            throw 2;
        }
        mesh = make_unique<mfem::Mesh>(imesh, 1, 1);
        imesh.close();
    }
    else if (nDimensions == 3)
    {
        mesh = make_unique<mfem::Mesh>(
                   N[0], N[1], N[2], mfem::Element::HEXAHEDRON, 1,
                   h(0) * N[0], h(1) * N[1], h(2) * N[2]);
    }
    else
    {
        mesh = make_unique<mfem::Mesh>(
                   N[0], N[1], mfem::Element::QUADRILATERAL, 1,
                   h(0) * N[0], h(1) * N[1]);
    }

    if (metis_partition)
    {
        auto elem_elem = TableToSparse(mesh->ElementToElementTable());
        auto part = Partition(elem_elem, num_procs,  proc_part_ubal);

        mfem::Array<int> partition(part.data(), part.size());

        pmesh_  = new mfem::ParMesh(comm, *mesh, partition);

        assert(partition.Max() + 1 == num_procs);
    }
    else
    {
        int num_procs_x = static_cast<int>(std::sqrt(num_procs) + 0.5);
        while (num_procs % num_procs_x)
            num_procs_x -= 1;

        num_procs_xyz_.resize(nDimensions);
        num_procs_xyz_[0] = num_procs_x;
        num_procs_xyz_[1] = num_procs / num_procs_x;
        if (nDimensions == 3)
            num_procs_xyz_[2] = 1;

        int nparts = 1;
        for (int d = 0; d < nDimensions; d++)
            nparts *= num_procs_xyz_[d];
        assert(nparts == num_procs);

        int* cart_part = mesh->CartesianPartitioning(num_procs_xyz_.data());
        pmesh_  = new mfem::ParMesh(comm, *mesh, cart_part);
        delete [] cart_part;
    }

    // Free the serial mesh
    mesh.reset();

    if (nDimensions == 3)
        pmesh_->ReorientTetMesh();

    // this should probably be in a different method
    Lx = N[0] * h(0);
    Ly = N[1] * h(1);
    Lz = N[2] * h(2);
    Hx = coarsening_factor[0] * h(0);
    Hy = coarsening_factor[1] * h(1);
    Hz = 1.0;
    if (nDimensions == 3)
        Hz = coarsening_factor[2] * h(2);
    source_coeff_ = new GCoefficient(Lx, Ly, Lz, Hx, Hy, Hz);
}

SPE10Problem::~SPE10Problem()
{
    InversePermeabilityFunction::ClearMemory();
    delete source_coeff_;
    delete kinv_;
    delete pmesh_;
}

void InversePermeabilityFunction::SetNumberCells(int Nx_, int Ny_, int Nz_)
{
    Nx = Nx_;
    Ny = Ny_;
    Nz = Nz_;
}

void InversePermeabilityFunction::SetMeshSizes(double hx_, double hy_,
                                               double hz_)
{
    hx = hx_;
    hy = hy_;
    hz = hz_;
}

void InversePermeabilityFunction::Set2DSlice(SliceOrientation o, int npos_ )
{
    orientation = o;
    npos = npos_;
}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName)
{
    std::ifstream buffer(fileName.c_str());

    if (!buffer.is_open())
    {
        std::cout << "Error in opening file " << fileName << std::endl;
        mfem::mfem_error("File does not exist");
    }

    std::stringstream permfile;
    permfile << buffer.rdbuf();

    inversePermeability = new double [3 * Nx * Ny * Nz];
    double* ip = inversePermeability;
    std::string tmp;

    for (int l = 0; l < 3; l++)
    {
        for (int k = 0; k < Nz; k++)
        {
            for (int j = 0; j < Ny; j++)
            {
                for (int i = 0; i < Nx; i++)
                {
                    tmp.clear();
                    permfile >> tmp;
                    *ip++ = 1. / (std::stod(tmp));
                }
                for (int i = 0; i < 60 - Nx; i++)
                    permfile >> tmp; // skip unneeded part
            }
            for (int j = 0; j < 220 - Ny; j++)
                for (int i = 0; i < 60; i++)
                    permfile >> tmp;  // skip unneeded part
        }

        if (l < 2) // if not processing Kz, skip unneeded part
            for (int k = 0; k < 85 - Nz; k++)
                for (int j = 0; j < 220; j++)
                    for (int i = 0; i < 60; i++)
                        permfile >> tmp;
    }

}

void InversePermeabilityFunction::ReadPermeabilityFile(const std::string& fileName,
                                                       MPI_Comm comm)
{
    int num_procs, myid;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);

    mfem::StopWatch chrono;

    chrono.Start();
    if (myid == 0)
        ReadPermeabilityFile(fileName);
    else
        inversePermeability = new double [3 * Nx * Ny * Nz];
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability file read in " << chrono.RealTime() << ".s \n";

    chrono.Clear();

    chrono.Start();
    MPI_Bcast(inversePermeability, 3 * Nx * Ny * Nz, MPI_DOUBLE, 0, comm);
    chrono.Stop();

    if (myid == 0)
        std::cout << "Permeability field distributed in " << chrono.RealTime() << ".s \n";

}

void InversePermeabilityFunction::InversePermeability(const mfem::Vector& x,
                                                      mfem::Vector& val)
{
    val.SetSize(x.Size());

    unsigned int i = 0, j = 0, k = 0;

    switch (orientation)
    {
        case NONE:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case XY:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = npos;
            break;
        case XZ:
            i = Nx - 1 - (int)floor(x[0] / hx / (1. + 3e-16));
            j = npos;
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        case YZ:
            i = npos;
            j = (int)floor(x[1] / hy / (1. + 3e-16));
            k = Nz - 1 - (int)floor(x[2] / hz / (1. + 3e-16));
            break;
        default:
            mfem::mfem_error("InversePermeabilityFunction::InversePermeability");
    }

    val[0] = inversePermeability[Ny * Nx * k + Nx * j + i];
    val[1] = inversePermeability[Ny * Nx * k + Nx * j + i + Nx * Ny * Nz];

    if (orientation == NONE)
        val[2] = inversePermeability[Ny * Nx * k + Nx * j + i + 2 * Nx * Ny * Nz];

}

double InversePermeabilityFunction::InvNorm2(const mfem::Vector& x)
{
    mfem::Vector val(3);
    InversePermeability(x, val);
    return 1. / val.Norml2();
}

void InversePermeabilityFunction::ClearMemory()
{
    delete[] inversePermeability;
}

int InversePermeabilityFunction::Nx(60);
int InversePermeabilityFunction::Ny(220);
int InversePermeabilityFunction::Nz(85);
double InversePermeabilityFunction::hx(20);
double InversePermeabilityFunction::hy(10);
double InversePermeabilityFunction::hz(2);
double* InversePermeabilityFunction::inversePermeability(NULL);
InversePermeabilityFunction::SliceOrientation InversePermeabilityFunction::orientation(
    InversePermeabilityFunction::NONE );
int InversePermeabilityFunction::npos(-1);

void FiniteVolumeMassIntegrator::AssembleElementMatrix(
    const mfem::FiniteElement& el,
    mfem::ElementTransformation& Trans,
    mfem::DenseMatrix& elmat)
{
    int dim = el.GetDim();
    int ndof = el.GetDof();
    elmat.SetSize(ndof);
    elmat = 0.0;

    mq.SetSize(dim);

    int order = 1;
    const mfem::IntegrationRule* ir = &mfem::IntRules.Get(el.GetGeomType(), order);

    MFEM_ASSERT(ir->GetNPoints() == 1, "Only implemented for piecewise "
                "constants!");

    int p = 0;
    const mfem::IntegrationPoint& ip = ir->IntPoint(p);

    if (VQ)
    {
        vq.SetSize(dim);
        VQ->Eval(vq, Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = vq(i);
    }
    else if (Q)
    {
        sq = Q->Eval(Trans, ip);
        for (int i = 0; i < dim; i++)
            mq(i, i) = sq;
    }
    else if (MQ)
        MQ->Eval(mq, Trans, ip);
    else
    {
        for (int i = 0; i < dim; i++)
            mq(i, i) = 1.0;
    }

    // Compute face area of each face
    mfem::DenseMatrix vshape;
    vshape.SetSize(ndof, dim);
    Trans.SetIntPoint(&ip);
    el.CalcVShape(Trans, vshape);
    vshape *= 2.;

    mfem::DenseMatrix vshapeT(vshape, 't');
    mfem::DenseMatrix tmp(ndof);
    Mult(vshape, vshapeT, tmp);

    mfem::Vector FaceAreaSquareInv(ndof);
    tmp.GetDiag(FaceAreaSquareInv);
    mfem::Vector FaceArea(ndof);

    for (int i = 0; i < ndof; i++)
        FaceArea(i) = 1. / std::sqrt(FaceAreaSquareInv(i));

    vshape.LeftScaling(FaceArea);
    vshapeT.RightScaling(FaceArea);

    // Compute k_{ii}
    mfem::DenseMatrix nk(ndof, dim);
    Mult(vshape, mq, nk);

    mfem::DenseMatrix nkn(ndof);
    Mult(nk, vshapeT, nkn);

    // this is right for grid-aligned permeability, maybe not for full tensor?
    mfem::Vector k(ndof);
    nkn.GetDiag(k);

    // here assume the input is k^{-1};
    mfem::Vector mii(ndof);
    for (int i = 0; i < ndof; i++)
        // Trans.Weight()/FaceArea(i)=Volume/face area=h (for rectangular grid)
        mii(i) = (Trans.Weight() / FaceArea(i)) * k(i) / FaceArea(i) / 2;
    elmat.Diag(mii.GetData(), ndof);
}

} // namespace rs2001
