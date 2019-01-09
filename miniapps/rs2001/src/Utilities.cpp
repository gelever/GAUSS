#include "Utilities.hpp"

namespace rs2001
{
void VectorToField(const gauss::VectorView& vect, mfem::GridFunction& field)
{
    assert(vect.size() == field.Size());

    int size = vect.size();

    for (int i = 0; i < size; ++i)
    {
        field[i] = vect[i];
    }
}

gauss::Vector VectorToVector(const mfem::Vector& mfem_vector)
{
    gauss::Vector vect(mfem_vector.Size());

    std::copy_n(mfem_vector.GetData(), mfem_vector.Size(), std::begin(vect));

    return vect;
}

mfem::Vector VectorToVector(const gauss::VectorView& vector)
{
    mfem::Vector vect(vector.size());

    std::copy_n(std::begin(vector), vector.size(), vect.GetData());

    return vect;
}

gauss::SparseMatrix SparseToSparse(const mfem::SparseMatrix& sparse)
{
    const int height = sparse.Height();
    const int width = sparse.Width();
    const int nnz = sparse.NumNonZeroElems();

    std::vector<int> indptr(height + 1);
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz);

    std::copy_n(sparse.GetI(), height + 1, std::begin(indptr));
    std::copy_n(sparse.GetJ(), nnz, std::begin(indices));
    std::copy_n(sparse.GetData(), nnz, std::begin(data));

    return gauss::SparseMatrix(std::move(indptr), std::move(indices), std::move(data), height, width);
}

gauss::SparseMatrix TableToSparse(const mfem::Table& table)
{
    const int height = table.Size();
    const int width = table.Width();
    const int nnz = table.Size_of_connections();


    std::vector<int> indptr(height + 1);
    std::vector<int> indices(nnz);
    std::vector<double> data(nnz, 1.0);

    std::copy_n(table.GetI(), height + 1, std::begin(indptr));
    std::copy_n(table.GetJ(), nnz, std::begin(indices));

    return gauss::SparseMatrix(std::move(indptr), std::move(indices), std::move(data), height, width);
}

gauss::ParMatrix ParMatrixToParMatrix(const mfem::HypreParMatrix& mat)
{
    mfem::SparseMatrix mfem_diag;
    mfem::SparseMatrix mfem_offd;
    HYPRE_Int* mfem_map;

    mat.GetDiag(mfem_diag);
    mat.GetOffd(mfem_offd, mfem_map);

    gauss::SparseMatrix diag = SparseToSparse(mfem_diag);
    gauss::SparseMatrix offd = SparseToSparse(mfem_offd);

    int col_map_size = offd.Cols();
    std::vector<HYPRE_Int> col_map(mfem_map, mfem_map + col_map_size);

    std::vector<HYPRE_Int> row_starts(mat.RowPart(), mat.RowPart() + 2);
    std::vector<HYPRE_Int> col_starts(mat.ColPart(), mat.ColPart() + 2);

    row_starts.push_back(mat.M());
    col_starts.push_back(mat.N());

    MPI_Comm comm = mat.GetComm();

    return gauss::ParMatrix(comm, row_starts, col_starts,
                            std::move(diag), std::move(offd), std::move(col_map));
}

gauss::DenseMatrix DenseToDense(const mfem::DenseMatrix& dense)
{
    gauss::DenseMatrix output;
    DenseToDense(dense, output);
    return output;
}

void DenseToDense(const mfem::DenseMatrix& dense, gauss::DenseMatrix& output)
{
    int rows = dense.Height();
    int cols = dense.Width();

    output.SetSize(rows, cols);

    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            output(i, j) = dense(i, j);
        }
    }
}

mfem::DenseMatrix DenseToDense(const gauss::DenseMatrix& dense)
{
    mfem::DenseMatrix output;
    DenseToDense(dense, output);
    return output;
}

void DenseToDense(const gauss::DenseMatrix& dense, mfem::DenseMatrix& output)
{
    int rows = dense.Rows();
    int cols = dense.Cols();

    output.SetSize(rows, cols);

    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            output(i, j) = dense(i, j);
        }
    }
}

std::vector<int> MetisPart(const gauss::SparseMatrix& vertex_edge, int num_parts)
{
    gauss::SparseMatrix edge_vertex = vertex_edge.Transpose();
    gauss::SparseMatrix vertex_vertex = vertex_edge.Mult(edge_vertex);

    double ubal_tol = 2.0;

    return Partition(vertex_vertex, num_parts, ubal_tol);
}

void SetVisView(mfem::socketstream& vis_v, VisType view)
{
    switch (view)
    {
        case VisType::Angle:
        {
            vis_v << vis_flatten << vis_flatten;
            vis_v << vis_embiggen << vis_embiggen;
            vis_v << vis_embiggen << vis_embiggen;
            vis_v << vis_angle_view;
            vis_v << vis_mesh_blend;
            vis_v << vis_mesh_level_lines;
            vis_v << vis_cet_rainbow_colormap;
            vis_v << vis_colorbar;

            break;
        }
        case VisType::Top:
        {
            vis_v << vis_top_view;
            vis_v << vis_light;
            vis_v << vis_perspective;
            vis_v << vis_embiggen << vis_embiggen;
            vis_v << vis_embiggen << vis_embiggen;
            vis_v << vis_embiggen << vis_embiggen;
            vis_v << vis_colorbar;
            break;
        }
        case VisType::Big:
        {
            vis_v << vis_embiggen << vis_embiggen;
            break;
        }
        default: throw std::runtime_error("Invalid View Selected!");
    }
}

void VisSetup(MPI_Comm comm, mfem::socketstream& vis_v, mfem::ParGridFunction& field,
              mfem::ParMesh& pmesh,
              VisRange range, const std::string& title, const std::string& caption, bool show_log,
              VisType vis_type)
{
    const char vishost[] = "localhost";
    const int  visport   = 19916;
    vis_v.open(vishost, visport);
    vis_v.precision(8);

    vis_v << "parallel " << pmesh.GetNRanks() << " " << pmesh.GetMyRank() << "\n";
    vis_v << "solution\n" << pmesh << field;
    vis_v << "window_size 500 800\n";

    if (!title.empty())
    {
        vis_v << "window_title '" << title << "'\n";
    }

    // update value-range; keep mesh-extents fixed
    vis_v << "autoscale off\n";

    // update value-range; keep mesh-extents fixed
    vis_v << "valuerange " << range.first << " " << range.second << "\n";

    SetVisView(vis_v, vis_type);

    if (!caption.empty())
    {
        vis_v << "plot_caption '" << caption << "'\n";
    }

    if (show_log)
    {
        vis_v << vis_log_display;
    }

    //vis_v << vis_pause; // Press space to play!

    MPI_Barrier(comm);

    //vis_v << vis_screenshot;
    //MPI_Barrier(comm);
}

void VisUpdate(MPI_Comm comm, mfem::socketstream& vis_v, mfem::ParGridFunction& field,
               mfem::ParMesh& pmesh)
{
    vis_v << "parallel " << pmesh.GetNRanks() << " " << pmesh.GetMyRank() << "\n";
    vis_v << "solution\n" << pmesh << field;

    MPI_Barrier(comm);

    //vis_v << "keys S\n";         //Screenshot
    //MPI_Barrier(comm);
}


void Visualize(const mfem::Vector& sol, mfem::ParMesh& pmesh, mfem::ParGridFunction& field,
               VisRange vis_range, const std::string& title, int level, bool show_log,
               VisType vis_type)
{
    MPI_Comm comm = pmesh.GetComm();
    mfem::socketstream vis_v;
    std::string caption = "Level: " + std::to_string(level);

    field = sol;

    VisSetup(comm, vis_v, field, pmesh, vis_range, title, caption, show_log, vis_type);

    VisUpdate(comm, vis_v, field, pmesh);
}

VisRange GetVisRange(MPI_Comm comm, const gauss::VectorView& vect)
{
    double local_lo = Min(vect);
    double local_hi = Max(vect);

    double global_lo;
    double global_hi;

    MPI_Allreduce(&local_lo, &global_lo, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_hi, &global_hi, 1, MPI_DOUBLE, MPI_MAX, comm);

    return std::make_pair<double, double>(std::move(global_lo), std::move(global_hi));
}

VisRange GetVisRange(MPI_Comm comm, const mfem::Vector& vect)
{
    double local_lo = vect.Min();
    double local_hi = vect.Max();

    double global_lo;
    double global_hi;

    MPI_Allreduce(&local_lo, &global_lo, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&local_hi, &global_hi, 1, MPI_DOUBLE, MPI_MAX, comm);

    if (std::fabs(global_hi - global_lo) < 1e-15)
    {
        global_lo -= 0.0001;
        global_hi += 0.0001;
    }

    return std::make_pair<double, double>(std::move(global_lo), std::move(global_hi));
}

gauss::SparseMatrix GenerateBoundaryAttributeTable(const mfem::Mesh& mesh)
{
    int nedges = mesh.Dimension() == 2 ? mesh.GetNEdges() : mesh.GetNFaces();
    int nbdr = mesh.bdr_attributes.Max();
    int nbdr_edges = mesh.GetNBE();

    std::vector<int> indptr(nedges + 1, 0);
    std::vector<int> indices(nbdr_edges);
    std::vector<double> data(nbdr_edges, 1.0);

    for (int j = 0; j < nbdr_edges; j++)
    {
        int edge = mesh.GetBdrElementEdgeIndex(j);
        indptr[edge + 1] = mesh.GetBdrAttribute(j);
    }

    int count = 0;

    for (int j = 1; j <= nedges; j++)
    {
        if (indptr[j])
        {
            indices[count++] = indptr[j] - 1;
            indptr[j] = indptr[j - 1] + 1;
        }
        else
        {
            indptr[j] = indptr[j - 1];
        }
    }

    return gauss::SparseMatrix(std::move(indptr), std::move(indices), std::move(data),
                               nedges, nbdr);
}

gauss::SparseMatrix GenerateBndrVertex(const mfem::Mesh& mesh,
                                       const mfem::FiniteElementSpace& fespace)
{
    int num_vertices = mesh.GetNV();
    int num_bdr = mesh.bdr_attributes.Max();
    int nbdr_edges = mesh.GetNBE();

    mfem::Array<int> ess_dofs;
    mfem::Array<int> bdr_attr_is_ess(num_bdr);
    bdr_attr_is_ess = 0;

    gauss::CooMatrix bdr_vertex;

    for (int i = 0; i < num_bdr; ++i)
    {
        bdr_attr_is_ess[i] = 1;
        fespace.GetEssentialVDofs(bdr_attr_is_ess, ess_dofs);

        int size = ess_dofs.Size();

        for (int j = 0; j < size; ++j)
        {
            if (ess_dofs[j] != 0)
            {
                bdr_vertex.Add(i, j, 1.0);
            }
        }

        bdr_attr_is_ess[i] = 0;
    }

    return bdr_vertex.ToSparse();
}

std::vector<int> MetisPart(mfem::ParFiniteElementSpace& sigmafespace,
                           mfem::ParFiniteElementSpace& ufespace,
                           mfem::Array<double>& coarsening_factor)
{
    mfem::DiscreteLinearOperator DivOp(&sigmafespace, &ufespace);
    DivOp.AddDomainInterpolator(new mfem::DivergenceInterpolator);
    DivOp.Assemble();
    DivOp.Finalize();

    int metis_coarsening_factor = 1;
    for (const auto factor : coarsening_factor)
        metis_coarsening_factor *= factor;

    return gauss::PartitionAAT(SparseToSparse(DivOp.SpMat()), metis_coarsening_factor);
}

std::vector<int> CartPart(std::vector<int>& num_procs_xyz,
                          mfem::ParMesh& pmesh, mfem::Array<double>& coarsening_factor)
{
    const int nDimensions = num_procs_xyz.size();

    mfem::Array<int> nxyz(nDimensions);
    nxyz[0] = 60 / num_procs_xyz[0] / coarsening_factor[0];
    nxyz[1] = 220 / num_procs_xyz[1] / coarsening_factor[1];
    if (nDimensions == 3)
        nxyz[2] = 85 / num_procs_xyz[2] / coarsening_factor[2];

    for (int& i : nxyz)
    {
        i = std::max(1, i);
    }

    mfem::Array<int> cart_part(pmesh.CartesianPartitioning(nxyz.GetData()), pmesh.GetNE());
    cart_part.MakeDataOwner();

    return std::vector<int>(cart_part.GetData(), cart_part.GetData() + cart_part.Size());
}

void EliminateEssentialBC(gauss::GraphUpscale& upscale,
                          const gauss::SparseMatrix& bdr_attr_vertex,
                          const std::vector<int>& ess_bdr,
                          const gauss::BlockVector& x,
                          gauss::BlockVector& b)
{
    const auto& mm_0 = upscale.GetLevel(0).mixed_matrix;

    int num_bdr = ess_bdr.size();
    int num_vertices = bdr_attr_vertex.Cols();
    int num_levels = upscale.NumLevels();

    gauss::SparseMatrix DT_elim = mm_0.LocalD().Transpose();
    gauss::SparseMatrix W_elim;

    if (mm_0.CheckW())
    {
        W_elim = mm_0.LocalW();
        W_elim *= -1.0;
    }
    else
    {
        W_elim = gauss::SparseMatrix(std::vector<double>(num_vertices, 0.0));
    }

    std::vector<int> marker(num_vertices, 0);

    for (auto&& dof_i : upscale.GetLevel(0).edge_elim_dofs)
    {
        DT_elim.EliminateRow(dof_i);
    }

    for (int i = 0; i < num_bdr; ++i)
    {
        if (ess_bdr[i])
        {
            auto bdr_vertices = bdr_attr_vertex.GetIndices(i);

            for (auto&& vertex : bdr_vertices)
            {
                marker[vertex] = 1;
                b.GetBlock(1)[vertex] = x.GetBlock(1)[vertex];
            }

            W_elim.EliminateRowCol(marker);
            DT_elim.EliminateCol(marker, x.GetBlock(1), b.GetBlock(0));

            for (auto&& vertex : bdr_vertices)
            {
                marker[vertex] = 0;
            }
        }
    }

    W_elim *= -1.0;
    gauss::SparseMatrix D_elim = DT_elim.Transpose();

    std::vector<gauss::MixedMatrix> mm_hats;
    mm_hats.emplace_back(mm_0.GetElemM(), mm_0.GetElemDof(),
                         std::move(D_elim), std::move(W_elim),
                         mm_0.EdgeTrueEdge());


    // Coarsen eliminated hierarchy
    for (int i = 0; i < num_levels - 1; ++i)
    {
        auto& mm_prev = mm_hats[i];
        auto& mm_next = upscale.GetMatrix(i + 1);
        auto& gc_i = upscale.Coarsener(i);

        auto PT_i = gc_i.Pvertex().Transpose();
        auto W_elim = PT_i.Mult(mm_prev.LocalW()).Mult(gc_i.Pvertex());
        auto D_elim = PT_i.Mult(mm_prev.LocalD()).Mult(gc_i.Pedge());

        mm_hats.emplace_back(mm_next.GetElemM(), mm_next.GetElemDof(),
                             std::move(D_elim), std::move(W_elim), mm_next.EdgeTrueEdge());
    }

    // Regenerate solvers w/ modified matrices
    for (int i = 0; i < num_levels; ++i)
    {
        upscale.MakeSolver(i, mm_hats[i]);
    }

    upscale.SetOrthogonalize(false);
}

void EliminateEssentialBC(gauss::MixedMatrix& mm_0,
                          gauss::MixedMatrix& mm_1,
                          const gauss::GraphCoarsen& coarsener,
                          const gauss::SparseMatrix& bdr_attr_vertex,
                          const std::vector<int>& ess_bdr,
                          const gauss::BlockVector& x,
                          gauss::BlockVector& b)
{
    int num_bdr = ess_bdr.size();
    int num_vertices = bdr_attr_vertex.Cols();

    gauss::SparseMatrix DT_elim = mm_0.LocalD().Transpose();
    gauss::SparseMatrix W_elim;

    if (mm_0.CheckW())
    {
        W_elim = mm_0.LocalW();
        W_elim *= -1.0;
    }
    else
    {
        W_elim = gauss::SparseMatrix(std::vector<double>(mm_0.LocalD().Rows(), 0.0));
    }

    std::vector<int> marker(std::max(mm_0.Rows(), mm_0.Cols()), 0);

    for (int i = 0; i < num_bdr; ++i)
    {
        if (ess_bdr[i])
        {
            auto bdr_vertices = bdr_attr_vertex.GetIndices(i);

            for (auto&& vertex : bdr_vertices)
            {
                marker[vertex] = 1;
                b.GetBlock(1)[vertex] = x.GetBlock(1)[vertex];
            }

            W_elim.EliminateRowCol(marker);
            DT_elim.EliminateCol(marker, x.GetBlock(1), b.GetBlock(0));

            for (auto&& vertex : bdr_vertices)
            {
                marker[vertex] = 0;
            }
        }
    }

    W_elim *= -1.0;
    gauss::SparseMatrix D_elim = DT_elim.Transpose();

    gauss::MixedMatrix mm_0_new(mm_0.GetElemM(), mm_0.GetElemDof(),
                         std::move(D_elim), std::move(W_elim),
                         mm_0.EdgeTrueEdge());
    mm_0 = std::move(mm_0_new);
    mm_0.AssembleM();
    //swap(mm_0, mm_0_new);

    {
    auto PT_i = coarsener.Pvertex().Transpose();
    auto W_elim = PT_i.Mult(mm_0.LocalW()).Mult(coarsener.Pvertex());
    auto D_elim = PT_i.Mult(mm_0.LocalD()).Mult(coarsener.Pedge());

    gauss::MixedMatrix mm_1_new(mm_1.GetElemM(), mm_1.GetElemDof(),
            std::move(D_elim), std::move(W_elim), mm_1.EdgeTrueEdge());
    mm_1 = std::move(mm_1_new);
    mm_1.AssembleM();
    }

}

} // namespace rs2001

