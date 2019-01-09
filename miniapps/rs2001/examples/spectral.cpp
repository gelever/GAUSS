#include "Utilities.hpp"

using namespace rs2001;
using linalgcpp::operator<<;

gauss::SparseMatrix DofEdge(const gauss::SparseMatrix& elem_dof);
gauss::SparseMatrix ElemEdge(const gauss::SparseMatrix& elem_dof);
std::vector<gauss::DenseMatrix> ElemMatrices(mfem::ParBilinearForm& a,
                                             mfem::ParFiniteElementSpace& fespace);

gauss::MixedMatrix AssembleBlock(const gauss::SparseMatrix& elem_dof,
                                 gauss::SparseMatrix elem_edge,
                                 gauss::SparseMatrix dof_edge,
                                 mfem::ParBilinearForm& a,
                                 mfem::ParFiniteElementSpace& fespace);

int main(int argc, char* argv[])
{
    gauss::MpiSession mpi(argc, argv);

    int num_evects = 1;
    int num_refine = 0;
    bool rand_refine = false;
    int ess_bdr_select = 0;
    double coarse_factor = 25.0;
    double ubal = 1.1;

    linalgcpp::ArgParser arg_parser(argc, argv);
    arg_parser.Parse(num_evects, "--m", "Num Evects");
    arg_parser.Parse(num_refine, "--nr", "Num Refine");
    arg_parser.Parse(rand_refine, "--rr", "Random Refine");
    arg_parser.Parse(coarse_factor, "--cf", "Coarsening Factor");
    arg_parser.Parse(ubal, "--ub", "Unbalance Factor");
    arg_parser.Parse(ess_bdr_select, "--ess", "Select ess");

    if (!arg_parser.IsGood())
    {
        ParPrint(mpi.myid, arg_parser.ShowHelp());
        ParPrint(mpi.myid, arg_parser.ShowErrors());

        return EXIT_FAILURE;
    }

    ParPrint(mpi.myid, arg_parser.ShowOptions());


    int gen_edges = 1;
    int num_wide = 2;
    int num_high = 2;
    int num_max = std::max(num_wide, num_high);
    //mfem::Mesh mesh(num_wide, num_high, mfem::Element::Type::QUADRILATERAL, gen_edges);
    //mfem::Mesh mesh(num_wide, num_high, mfem::Element::Type::TRIANGLE, gen_edges);
    mfem::Mesh mesh("mesh.mesh");
    mfem::ParMesh pmesh(mpi.comm, mesh);

    for (int i = 0; i < num_refine; ++i)
    {
        if (rand_refine)
        {
            pmesh.RandomRefinement(0.15);
        }
        else
        {
            pmesh.UniformRefinement();
        }
    }

    int dim = pmesh.Dimension();

    mfem::H1_FECollection fec(1, dim);
    mfem::ParFiniteElementSpace fespace(&pmesh, &fec);
    mfem::ParGridFunction x(&fespace);

    std::vector<int> ess_bdr(pmesh.bdr_attributes.Max(), ess_bdr_select);
    //ess_bdr[0] = ess_bdr_select;
    //ess_bdr[1] = ess_bdr_select;
    //ess_bdr[2] = ess_bdr_select;
    //ess_bdr[3] = ess_bdr_select;

    mfem::Array<int> ess_bdr_mfem(ess_bdr.size());

    for (int i = 0; i < ess_bdr.size(); ++i)
    {
        ess_bdr_mfem[i] = ess_bdr[i];
    }

    mfem::Array<int> ess_tdof_list;
    if (pmesh.bdr_attributes.Size())
    {
        fespace.GetEssentialVDofs(ess_bdr_mfem, ess_tdof_list);
    }

    gauss::SparseMatrix bdr_attr_vertex = GenerateBndrVertex(pmesh, fespace);

    mfem::ConstantCoefficient one(1.0);

    mfem::ParBilinearForm a(&fespace);
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator(one));
    a.Assemble();

    std::vector<gauss::DenseMatrix> elem_matrices = ElemMatrices(a, fespace);

    mfem::ParLinearForm b(&fespace);
    b.AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
    b.Assemble();

    gauss::SparseMatrix elem_dof = TableToSparse(fespace.GetElementToDofTable());
    gauss::SparseMatrix dof_elem = elem_dof.Transpose();

    gauss::SparseMatrix elem_edge = ElemEdge(elem_dof);
    gauss::SparseMatrix dof_edge = DofEdge(elem_dof);

    gauss::MixedMatrix mm = AssembleBlock(elem_dof, elem_edge, dof_edge, a, fespace);
    mm.AssembleM();

    gauss::ParMatrix my_assemble = mm.ToPrimal();

    //mfem::Array<int> ess_tdof_list2;
    mfem::HypreParMatrix A;
    mfem::Vector B, X;
    fespace.GetEssentialTrueDofs(ess_bdr_mfem, ess_tdof_list);
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);


    if (ess_bdr_select == 0)
    {
        int size = mpi.myid == 0 ? 1 : 0;
        mfem::Array<int> elim(size);
        elim = 0;
        A.EliminateRowsCols(elim);

    }

    //gauss::ParMatrix mfem_assemble = ParMatrixToParMatrix(A);

    //gauss::ParMatrix dof_true_dof = ParMatrixToParMatrix(*fespace.Dof_TrueDof_Matrix());
    //gauss::ParMatrix true_dof_dof = dof_true_dof.Transpose();
    //gauss::ParMatrix my_assemble_true = linalgcpp::Mult(true_dof_dof, my_assemble, dof_true_dof);

    //mm.LocalW().ToDense().Print("my W " + std::to_string(mpi.myid), std::cout, 6, 2);
    //my_assemble.GetDiag().ToDense().Print("my A " + std::to_string(mpi.myid), std::cout, 6, 2);
    //my_assemble_true.GetDiag().ToDense().Print("my A true" + std::to_string(mpi.myid), std::cout, 6, 2);
    //mfem_assemble.GetDiag().ToDense().Print("mfem A " + std::to_string(mpi.myid), std::cout, 6, 2);


    const auto& edge_true_edge = mm.EdgeTrueEdge();
    auto vertex_edge = mm.LocalD();
    vertex_edge = 1.0;

    gauss::Vector rhs(b.Size());
    for (int i = 0; i < rhs.size(); ++i)
    {
        rhs[i] = b[i];
    }

    //int nxyz[] = {coarse_factor, coarse_factor, coarse_factor};
    //int nxyz[] = {coarse_factor, 1, coarse_factor};
    int nxyz[] = {(int)coarse_factor, 1, 1};
    auto elem_part_int = pmesh.CartesianPartitioning(nxyz);
    std::vector<int> elem_part(pmesh.GetNE());
    std::copy(elem_part_int, elem_part_int + pmesh.GetNE(), std::begin(elem_part));
    delete[] elem_part_int;
    int num_dof = dof_elem.Rows();
    std::vector<int> part(num_dof);
    {
        auto agg_elem = linalgcpp::MakeSetEntity(elem_part);
        auto agg_dof = agg_elem.Mult(elem_dof);
        auto dof_agg = agg_dof.Transpose();
        //dof_agg.ToDense().Print("Dof Agg elem:", std::cout, 4, 0);

        for (int i = 0;i < num_dof; ++i)
        {
            linalgcpp::linalgcpp_verify(dof_agg.RowSize(i) > 0);
            part[i] = dof_agg.GetIndices(i)[0];
        }
    }


    //auto dof_dof = dof_elem.Mult(elem_dof);
    //const auto& part = gauss::PartitionAAT(vertex_edge, 2.0, 1.0, true);
    //int num_parts = std::max(2.0, (dof_dof.Rows() / coarse_factor) + 0.5);
    //printf("Part Parts: %d\n", num_parts);
    //const auto& part = linalgcpp::Partition(dof_dof, num_parts, ubal);
    //std::vector<int> part{0, 1, 1, 2, 2, 1, 2, 2, 3};

    //std::cout << "Part: " << part;


    gauss::Graph graph(vertex_edge, edge_true_edge, part);
    gauss::GraphTopology gt(graph);
    std::cout.flush();

    //graph.vertex_edge_local_.ToDense().Print("vertex_edge:", std::cout, 2, 0);
    //graph.vertex_edge_local_.Transpose().ToDense().Print("edge_vertex:", std::cout, 2, 0);
    //gt.agg_vertex_local_.ToDense().Print("agg_vertex:", std::cout, 2, 0);
    //gt.agg_edge_local_.ToDense().Print("agg_edge:", std::cout, 2, 0);
    //gt.face_edge_.GetDiag().ToDense().Print("face_edge:", std::cout, 2, 0);
    //gt.agg_face_local_.Transpose().ToDense().Print("face_agg:", std::cout, 2, 0);

    auto vertex_agg = gt.agg_edge_local_.Mult(graph.vertex_edge_local_.Transpose()).Transpose();

    bool show_mesh = true;
    if (show_mesh)
    {
        VisRange range{-1.0, 1.0};
        mfem::ParGridFunction vertex_gf(&fespace);

        bool show_log = false;
        VisType vis_type = VisType::Big;

        mfem::Vector sol(vertex_gf.Size());
        sol = 0.0;
        Visualize(sol, pmesh, vertex_gf,
                  range, "mesh", 0, show_log, vis_type);
        //std::cout << "part:" << part;
    }

    int bad_part = 0;
    for (int i = 0; i < vertex_agg.Rows(); ++i)
    {
        if (vertex_agg.RowSize(i) == 0)
        {
            std::cout << "Bad part: " << i <<"\n";
            bad_part++;
        }
    }

    if (bad_part != 0)
    {
        ParPrint(mpi.myid, printf("Bad Part! \n"));
        //throw std::runtime_error("Bad Part!: " + std::to_string(bad_part));
    }

    printf("VE: %d %d\n", vertex_edge.Rows(), vertex_edge.Cols());

    //auto agg_edge = gt.agg_vertex_local_.Mult(dof_edge);

    //agg_edge.ToDense().Print("AggEdge", std::cout, 2, 0);

    gauss::SpectralPair sp(1.0, num_evects);
    gauss::GraphSpace gs = gauss::FineGraphSpace(graph);
    gauss::Vector constant_rep(mm.LocalD().Rows(), 1.0 / std::sqrt(mm.LocalD().Rows()));
    gauss::GraphCoarsen gc(graph, gs, mm, constant_rep, sp);

    auto QT = gc.Qedge().Transpose();
    auto PT = gc.Pedge().Transpose();
    auto QTP = QT.Mult(gc.Pedge());
    auto PTQ = PT.Mult(gc.Qedge());

    //gc.Qedge().ToDense().Print("Qedge:", std::cout, 6, 2);
    //gc.Pedge().ToDense().Print("Pedge:", std::cout, 6, 2);
    //gc.Pvertex().ToDense().Print("Pvertex:", std::cout, 6, 2);
    //QTP.ToDense().Print("QTP:", std::cout, 6, 2);
    //PTQ.ToDense().Print("PTQ:", std::cout, 6, 2);

    //auto PD = gc.Pvertex().Transpose().Mult(mm.LocalD());
    //auto DP = mm.LocalD().Mult(gc.Pedge());
    //auto DPQ = DP.Mult(gc.Qedge().Transpose());
    //auto PPD = gc.Pvertex().Mult(gc.Pvertex().Transpose()).Mult(mm.LocalD());
    //mm.LocalD().ToDense().Print("D :", std::cout, 6, 2);
    //PD.ToDense().Print("P D :", std::cout, 6, 2);
    //DP.ToDense().Print("D P:", std::cout, 6, 2);
    //gc.Qedge().Transpose().ToDense().Print("QT:", std::cout, 6, 2);
    //DPQ.ToDense().Print("DPQ:", std::cout, 6, 2);
    //PPD.ToDense().Print("PPD:", std::cout, 6, 2);

    //gauss::Vector edge_vect(mm.LocalD().Cols());
    //linalgcpp::Randomize(edge_vect, -1.0, 1.0);
    //auto test = mm.LocalD().Mult(gc.Pedge().Mult(gc.Qedge().MultAT(edge_vect)));
    //test.Print("DPQ sigma");
    //std::cout <<" TEST:" << test.Mult(constant_rep) << "\n";

    //mm.LocalD().Transpose().Mult(mm.LocalD()).ToDense().Print("DTD :", std::cout, 6, 2);

    //auto proj_diff = DPQ.ToDense() - PPD.ToDense();
    //proj_diff.Print("proj diff:", std::cout, 6, 2);
    //std::ofstream mesh_out("mesh.mesh");
    //pmesh.Print(mesh_out);

    //gc.Pvertex().Transpose().Mult(mm.LocalD()).Transpose().ToDense().Print("D P:", std::cout, 6, 2);


    //const gauss::GraphTopology& gt_f = gc.Topology();
    //gauss::GraphSpace gs_c = gc.BuildGraphSpace();
    //gauss::GraphTopology gt_c(gt_f, 4);

    //gauss::Vector constant_f(gc.Pvertex().Rows(), 1.0 / std::sqrt(gc.Pvertex().Rows()));
    //gauss::Vector constant_c = gc.Restrict(constant_f);

    gauss::MixedMatrix mm_c = gc.Coarsen(mm);
    mm_c.AssembleM();


    gauss::Vector ones(rhs.size(), 1.0 / std::sqrt(rhs.size()));
    const auto& M_ass = mm_c.GlobalM();
    const auto& P_edge = gc.Pedge();
    const auto& P_edge_T = P_edge.Transpose();
    const auto& M_rap = P_edge_T.Mult(mm.LocalM()).Mult(P_edge);

    //mm.LocalM().ToDense().Print("M fine", std::cout, 6, 2);
    //P_edge.ToDense().Print("P edge", std::cout, 8, 2);
    //M_rap.ToDense().Print("M rap", std::cout, 8, 2);
    //M_ass.GetDiag().ToDense().Print("M ass", std::cout, 8, 2);

    gauss::Vector u(M_rap.Rows());
    linalgcpp::Randomize(u, -1.0, 1.0);
    linalgcpp::Normalize(u);

    auto u_ass = M_ass.Mult(u);
    auto u_rap = M_rap.Mult(u);
    auto diff = (u_ass - u_rap).L2Norm() / u_rap.L2Norm();

    auto ones2(ones);
    ones2 = 1.0;
    auto DTone = mm.LocalD().MultAT(ones2).L2Norm();

    auto ones2_c = gc.Restrict(ones2);
    auto DTone_c = mm_c.LocalD().MultAT(ones2_c).L2Norm();

    ParPrint(mpi.myid, printf("D_c: %d %d DT 1: %.4e DT_c 1: %.4e\n",
                mm_c.LocalD().Rows(), mm_c.LocalD().Cols(),
                DTone, DTone_c));
    ParPrint(mpi.myid, printf("M_c: %d %d (M-assemble-M-rap) u: %.4e\n",
                mm_c.LocalM().Rows(), mm_c.LocalM().Cols(),
                diff));

    //mm_c.LocalM().ToDense().Print("M_c", std::cout, 8, 2);
    //mm_c.LocalD().ToDense().Print("D_c", std::cout, 8, 2);

    //gauss::GraphCoarsen gc_c(gt_c, gs_c, mm_c, constant_c, sp);
    //gauss::MixedMatrix mm_c_c = gc_c.Coarsen(mm_c);
    //mm_c_c.AssembleM();

    double fine_nnz = mm.GlobalNNZ();
    double coarse_nnz = mm_c.GlobalNNZ();

    double op_comp = (fine_nnz + coarse_nnz) / fine_nnz;

    ParPrint(mpi.myid, printf("Op Comp: %.8f\n", op_comp));

    gauss::BlockVector vertex_data(mm.Offsets());
    vertex_data = 0.0;

    gauss::BlockVector block_rhs(mm.Offsets());
    block_rhs.GetBlock(0) = 0.0;
    block_rhs.GetBlock(1) = rhs;

    if (ess_bdr_select == 0)
    {
        gauss::OrthoConstant(mpi.comm, block_rhs.GetBlock(1), ones);
    }

    std::vector<double> ess_vals(ess_bdr.size(), 0.0);

    for (int attr = 0; attr < ess_bdr.size(); ++attr)
    {
        if (ess_bdr[attr])
        {
            auto vertices = bdr_attr_vertex.GetIndices(attr);

            for (auto&& dof : vertices)
            {
                vertex_data.GetBlock(1)[dof] = ess_vals[attr];
                block_rhs.GetBlock(1)[dof] = ess_vals[attr];
            }
        }
    }

    printf("W: %d %d\n", mm.LocalW().Rows(), mm.LocalW().Cols());

    if (ess_bdr_select)
    {
    EliminateEssentialBC(mm, mm_c, gc, bdr_attr_vertex, ess_bdr,
                         vertex_data, block_rhs);
    }

    //mm_c.LocalM().ToDense().Print("elim M_c", std::cout, 8, 2);
    //mm_c.LocalD().ToDense().Print("elim D_c", std::cout, 8, 2);

    //vertex_data.Print("vertex data");
    //block_rhs.Print("block rhs");

    //mm_c.LocalM().ToDense().Print("M_c", std::cout, 6, 2);
    //mm_c.LocalD().ToDense().Print("D_c", std::cout, 6, 2);
    //mm_c.LocalM().Print("M_c", std::cout);
    //mm_c.LocalD().Print("D_c", std::cout);

        for (int k = 0; k < block_rhs.GetBlock(1).size(); ++k)
        {
            B[k] = block_rhs.GetBlock(1)[k];
        }

    mfem::HypreBoomerAMG amg(A);
    mfem::HyprePCG pcg(A);
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(200);
    amg.SetPrintLevel(0);
    pcg.SetPrintLevel(0);
    pcg.SetPreconditioner(amg);
    //pcg.Mult(B, X);

    a.RecoverFEMSolution(X, b, x);
    //printf("b:");
    //b.Print();
    //printf("x:");
    //x.Print();


    gauss::MinresBlockSolver minres(mm);
    gauss::MinresBlockSolver minres_c(mm_c);


    //minres_c.SetRelTol(5e-7);
    //minres_c.SetAbsTol(5e-7);
    //minres_c.SetMaxIter(200000);
    //gauss::MinresBlockSolver minres_c_c(mm_c_c);

    gauss::BlockVector rhs_c = gc.Restrict(block_rhs);
    //gauss::BlockVector rhs_c_c = gc_c.Restrict(rhs_c);

    gauss::BlockVector sol = minres.Mult(block_rhs);
    gauss::BlockVector sol_c = minres_c.Mult(rhs_c);
    //gauss::BlockVector sol_c_c = minres_c_c.Mult(rhs_c_c);

    gauss::BlockVector sol_up = gc.Interpolate(sol_c);
    //gauss::BlockVector sol_up_up = gc.Interpolate(gc_c.Interpolate(sol_c_c));

    sol.GetBlock(1) *= -1.0;
    sol_up.GetBlock(1) *= -1.0;

    if (ess_bdr_select == 0)
    {
        gauss::OrthoConstant(mpi.comm, sol.GetBlock(1), ones);
        gauss::OrthoConstant(mpi.comm, sol_up.GetBlock(1), ones);
    }
    //sol_up_up.GetBlock(1) *= -1.0;

    auto info = gauss::ComputeErrors(mpi.comm, mm.LocalM(), mm.LocalD(), sol_up, sol);
    //auto info2 = gauss::ComputeErrors(mpi.comm, mm.LocalM(), mm.LocalD(), sol_up_up, sol);
    info.push_back(op_comp);
    //info2.push_back(op_comp);

    ParPrint(mpi.myid, gauss::ShowErrors(info));
    //ParPrint(mpi.myid, gauss::ShowErrors(info2));

    //sol_c.GetBlock(1).Print("Sol Coarse:");
    //sol.GetBlock(1).Print("Sol Fine:");
    //sol_up.GetBlock(1).Print("Sol Upscaled:");

    //elem_dof.ToDense().Print("elem dof:", std::cout, 2, 0);
    //dof_elem.ToDense().Print("dof elem:", std::cout, 2, 0);
    //elem_edge.ToDense().Print("elem edge:", std::cout, 2, 0);
    //dof_edge.ToDense().Print("dof edge:", std::cout, 2, 0);

    //for (auto&& elem_i : elem_matrices)
    //{
    //    elem_i.Print("Elem_i", std::cout, 8, 4);
    //}

    //for (int i = 0; i < x.Size(); ++i)
    //{
    //    x[i] = mpi.myid * mpi.num_procs + i + 1.0;
    //}

    bool visualization = true;
    if (visualization)
    {
        VisRange range = GetVisRange(mpi.comm, sol.GetBlock(1));
        VisRange range_c = GetVisRange(mpi.comm, sol_up.GetBlock(1));
        mfem::ParGridFunction vertex_gf(&fespace);

        bool show_log = false;
        VisType vis_type = VisType::Big;

        //Visualize(x, pmesh, vertex_gf, range, "mfem x", 0, show_log, vis_type);
        Visualize(VectorToVector(sol.GetBlock(1)), pmesh, vertex_gf,
                  range, "fine x", 0, show_log, vis_type);
        Visualize(VectorToVector(sol_up.GetBlock(1)), pmesh, vertex_gf,
                  range, "upscaled x; m:" + std::to_string(num_evects), 1, show_log, vis_type);
        //Visualize(VectorToVector(sol_up_up.GetBlock(1)), pmesh, vertex_gf,
                  //range, "upupscaled x", 2, show_log, vis_type);
    }

    return EXIT_SUCCESS;
}

gauss::SparseMatrix DofEdge(const gauss::SparseMatrix& elem_dof)
{
    int num_elem = elem_dof.Rows();
    int num_dof = elem_dof.Cols();
    int size = elem_dof.RowSize(0) - 1;
    int num_edge = size * num_elem;

    gauss::CooMatrix dof_edge(num_dof, num_edge);

    for (int i = 0; i < num_elem; ++i)
    {
        std::vector<int> dofs = elem_dof.GetIndices(i);

        for (int j = 0; j < size; ++j)
        {
            int edge_i = (i * size) + j;

            for (auto&& dof_i : dofs)
            {
                dof_edge.Add(dof_i, edge_i, 1.0);
            }
        }
    }

    return dof_edge.ToSparse();
}

gauss::SparseMatrix ElemEdge(const gauss::SparseMatrix& elem_dof)
{
    int num_elem = elem_dof.Rows();
    int num_dof = elem_dof.RowSize(0) - 1;
    int num_edge = num_dof * num_elem;

    gauss::CooMatrix elem_edge(num_elem, num_edge);

    for (int i = 0; i < num_elem; ++i)
    {
        for (int j = 0; j < num_dof; ++j)
        {
            int edge_i = (i * num_dof) + j;
            elem_edge.Add(i, edge_i, 1.0);
        }
    }

    return elem_edge.ToSparse();
}

std::vector<gauss::DenseMatrix> ElemMatrices(mfem::ParBilinearForm& a,
                                             mfem::ParFiniteElementSpace& fespace)
{
    int num_elem = fespace.GetNE();

    std::vector<gauss::DenseMatrix> elem_mats(num_elem);
    mfem::DenseMatrix buffer(3, 3);

    a.ComputeElementMatrices();

    for (int i = 0; i < num_elem; ++i)
    {
        a.ComputeElementMatrix(i, buffer);
        elem_mats[i] = DenseToDense(buffer);
    }

    return elem_mats;
}

gauss::MixedMatrix AssembleBlock(const gauss::SparseMatrix& elem_dof,
                                 gauss::SparseMatrix elem_edge,
                                 gauss::SparseMatrix dof_edge,
                                 mfem::ParBilinearForm& a,
                                 mfem::ParFiniteElementSpace& fespace)
{
    int num_elem = elem_dof.Rows();
    int num_dofs = elem_dof.Cols();
    int num_edge = elem_edge.Cols();

    std::vector<gauss::DenseMatrix> elem_mats(num_elem);

    int size = elem_dof.RowSize(0);

    mfem::DenseMatrix elem_mat(size, size);
    gauss::DenseMatrix buffer(size, size);

    gauss::CooMatrix D_coo(num_dofs, num_edge);
    std::vector<gauss::DenseMatrix> M_elem(num_dofs, {size - 1, size - 1});

    a.ComputeElementMatrices();

    linalgcpp::EigenSolver eigen;
    linalgcpp::EigenPair eigen_pair;

    std::vector<double> evals(num_edge, 0.0);
    //mfem::Array<int> vertices;

    for (int i = 0; i < num_elem; ++i)
    {
        auto dofs = elem_dof.GetIndices(i);
        auto edges = elem_edge.GetIndices(i);

        int num_dofs = dofs.size();
        int num_edges = edges.size();

        //fespace.GetElementVertices(i, vertices);
        //std::cout << "Dofs:" << dofs;
        //std::cout << "Vertices:\n";
        //vertices.Print();

        linalgcpp::linalgcpp_verify(num_dofs == num_edges + 1,
                         "Edge dof mismatch!");

        a.ComputeElementMatrix(i, elem_mat);
        DenseToDense(elem_mat, buffer);

        eigen.Solve(buffer, 1.0, buffer.Cols(), eigen_pair);
        eigen_pair.second.GetCol(1, num_dofs, buffer);

        for (int j = 0; j < num_edges; ++j)
        {
            evals[edges[j]] = eigen_pair.first[j + 1];
        }

        D_coo.Add(dofs, edges, buffer);
    }

    for (int i = 0; i < num_dofs; ++i)
    {
        auto edges = dof_edge.GetIndices(i);
        int num_edges = edges.size();

        M_elem[i].SetSize(num_edges, num_edges);

        for (int j = 0; j < num_edges; ++j)
        {
            M_elem[i](j, j) = 1.0 / (size * evals[edges[j]]);
        }
    }

    gauss::SparseMatrix D = D_coo.ToSparse();

    gauss::SparseMatrix ident = linalgcpp::SparseIdentity(num_edge);
    gauss::ParMatrix edge_true_edge(fespace.GetComm(), std::move(ident));

    return gauss::MixedMatrix(std::move(M_elem), std::move(dof_edge), std::move(D),
                              {}, std::move(edge_true_edge));
}
