/*
This program uses the sparse matrix method of calculating the real-frequency spectral functions from imaginary-time Green's functions
Copyright (C) 2025 Alexandru Golic (soricib@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "spm.h"


spm::Grid spm::generate_centrosymmetric_grid(Vector omegas_d, Vector domegas, int n_taus, double beta_d, double recursion_tolerance) {
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(100));
    const auto start{std::chrono::steady_clock::now()};
    double tol = -1;
    PScalar beta = static_cast<PScalar>(beta_d);
    PScalar sqrt2 = mpfr::sqrt(static_cast<PScalar>(2));
    PScalar sqrt2inv = static_cast<PScalar>(1)/sqrt2;
    int N = n_taus;
    int M = omegas_d.size();
    int p = N/2;
    int a = N % 2;
    int q = M/2;
    int b = M % 2;
    logger::log->info("Generating block kernel...");
    PVector taus = symmetric_linspace(N, beta/2, beta/2);
    PVector omegas = symmetric_linspace(M, static_cast<PScalar>(omegas_d(omegas_d.size() - 1)));
    PMatrix A = PMatrix::Zero(N, M);
    for (int i = 0; i < N; i++) {
        for (int j = 0 ; j < M ; j++) {
            A(i, j) = -mpfr::exp(-taus(i)*omegas(j))/(1 + mpfr::exp(-beta*omegas(j)));
        }
    }
    PMatrix Ip = PMatrix::Identity(p, p);
    PMatrix Iq = PMatrix::Identity(q, q);
    PMatrix Jp = get_j_matrix(p);
    PMatrix Jq = get_j_matrix(q);
    PMatrix U0 = PMatrix::Zero(N, N);
    U0.block(0, 0, p, p) = Ip;
    U0.block(0, p+a, p, p) = Ip;
    U0.block(p+a, 0, p, p) = Jp;
    U0.block(p+a, p+a, p, p) = -Jp;
    if (a == 1) {
        U0(p+a-1, p+a-1) = sqrt2;
    }
    U0 *= sqrt2inv;
    PMatrix V0 = PMatrix::Zero(M, M);
    V0.block(0, 0, q, q) = Iq;
    V0.block(0, q+b, q, q) = Iq;
    V0.block(q+b,0,q,q) = Jq;
    V0.block(q+b,q+b,q,q) = -Jq;
    if (b == 1) {
        V0(q+b-1, q+b-1) = sqrt2;
    }
    V0 *= sqrt2inv;
    //std::cout << A << "\n\n";
    //std::cout << U0 << "\n\n";
    //std::cout << V0 << "\n\n";
    //std::cout << A_block << "\n\n";
    PMatrix B1 = PMatrix::Zero(p+a, q+b);
    PMatrix B2 = PMatrix::Zero(p, q);
    for (int i = 0 ; i < p ; i++) {
        for (int j = 0 ; j < q ; j++) {
            B1(i,j) = mpfr::sinh(taus(i)*omegas(j))*mpfr::tanh(omegas(j)*beta/2) - mpfr::cosh(taus(i)*omegas(j));
            B2(i, j) = mpfr::sinh(taus(i)*omegas(j)) - mpfr::cosh(taus(i)*omegas(j))*mpfr::tanh(omegas(j)*beta/2);
        }
    }
    if (a == 1) {
        for (int j = 0 ; j < q ; j++) {
            B1(p+a-1, j) = -1/(sqrt2*mpfr::cosh(omegas(j)*beta/2));
        }
    }
    if (b == 1) {
        for (int i = 0 ; i < p ; i++) {
            B1(i, q+b-1) = -sqrt2inv;
        }
    }
    if (a == 1 && b == 1) {
        B1(p+a-1, q+b-1) = static_cast<PScalar>(-0.5);
    }
    PMatrix B = PMatrix::Zero(N, M);
    B.block(0, 0, p+a, q+b) = B1;
    B.block(p+a, q+b, p, q) = B2;
    //std::cout << A << "\n\n";
    //std::cout << (A - U0*B*V0.transpose()) << "\n\n";
    //std::cout << B1 << "\n\n";
    //std::cout << B2 << "\n\n";
    PSVD svd_1, svd_2;
    logger::log->info("Starting both SVDs...");


    #pragma omp parallel num_threads(2)
    {
        mpfr::mpreal::set_default_prec(mpfr::digits2bits(100));
        int tid = omp_get_thread_num();
        if (tid == 0) {
            logger::log->info("Starting SVD decomposition for upper block");
            svd_1 = recursive_svd_mp(B1, tol);
            logger::log->info("Upper block done");
        } else if (tid == 1) {
            logger::log->info("Starting SVD decomposition for lower block");
            svd_2 = recursive_svd_mp(B2, tol);
            logger::log->info("Lower block done");
        }
    }

    //svd_1 = recursive_svd(B1, tol);
    //svd_2 = recursive_svd(B2, tol);
    logger::log->info("Combining SVDs...");
    //std::cout << "B1 - B1 svd :\n" << "\n";
    //std::cout << (B1 - svd_1.U * svd_1.SVs.asDiagonal().toDenseMatrix() * svd_1.V.transpose()) << "\n\n";
    PMatrix sigma_tilde = PMatrix::Zero(N, M);
    //std::cout << sigma_tilde << "\n\n";
    //std::cout << svd_1.SVs.asDiagonal().toDenseMatrix() << "\n\n";
    //std::cout << svd_2.SVs.asDiagonal().toDenseMatrix() << "\n\n";
    int l = std::min(p, q);
    int lc = std::min(p+a, q+b);
    //fmt::print("l : {}, lc : {}\n", l, lc);
    sigma_tilde.block(0, 0, lc, lc) = svd_1.SVs.asDiagonal().toDenseMatrix();
    sigma_tilde.block(p+a, q+b, l, l) = svd_2.SVs.asDiagonal().toDenseMatrix();
    //std::cout << sigma_tilde << "\n\n";
    PMatrix U_c = PMatrix::Zero(N, N);
    U_c.block(0, 0, p+a, p+a) = svd_1.U;
    U_c.block(p+a, p+a, p, p) = svd_2.U;
    PMatrix V_c = PMatrix::Zero(M, M);
    V_c.block(0, 0, q+b, q+b) = svd_1.V;
    V_c.block(q+b, q+b, q, q) = svd_2.V;

    //SVD svd_direct = recursive_svd(A, tol);
    //svd_1 = recursive_svd(B1, tol);
    //svd_2 = recursive_svd(B2, tol);
    //std::cout << (A - svd_direct.U * svd_direct.SVs.asDiagonal().toDenseMatrix()*svd_direct.V.transpose()) << "\n\n";
    //std::cout << (A - U0*U_c*sigma_tilde*V_c.transpose()*V0.transpose()) << "\n\n";
    //std::cout << (B1 - svd_1.U * svd_1.SVs.asDiagonal().toDenseMatrix() * svd_1.V.transpose()) << "\n\n";
    //std::cout << (B2 - svd_2.U * svd_2.SVs.asDiagonal().toDenseMatrix() * svd_2.V.transpose()) << "\n\n";
    //std::cout << A << "\n\n";
    int gap = p + a - q - b;
    Eigen::PermutationMatrix<Eigen::Dynamic> U_p;
    Eigen::PermutationMatrix<Eigen::Dynamic> V_p;
    Eigen::VectorXi perm_U = Eigen::VectorXi::LinSpaced(N, 0, N-1);
    Eigen::VectorXi perm_V = Eigen::VectorXi::LinSpaced(M, 0, M-1);
    if (gap > 0) {
        int perm_dim = gap + p;
        for (int i = 0; i < perm_dim; i++) {
            perm_U(q + b + (perm_dim + i - gap) % perm_dim) = i + q + b;
        }
    }
    if (gap < 0) {
        gap = -gap;
        int perm_dim = gap + q;
        for (int j = 0; j < perm_dim; j++) {
            perm_V(p + a + (perm_dim + j - gap) % perm_dim) = j + p + a;
        }
    }
    U_p.indices() = perm_U;
    V_p.indices() = perm_V;
    //std::cout << U_p << "\n\n";
    //std::cout << U_p2.toDenseMatrix() << "\n\n";
    //std::cout << V_p << "\n\n";
    //std::cout << V_p2.toDenseMatrix() << "\n\n";
    PMatrix sigma_prime = U_p.transpose()*sigma_tilde*V_p;
    logger::log->info("Done.");
    logger::log->info("Sorting SVs...");
    //std::cout << sigma_prime << "\n\n";
    int L = std::min(N, M);
    std::vector<std::pair<PScalar, int>> data(L);
    PVector SVs = PVector::Zero(L);
    for (int i = 0; i < L; i++) {
        data[i] = {sigma_prime(i, i), i};
        //std::cout << data[i].first << ", " << data[i].second << std::endl;
    }
    std::ranges::sort(data.begin(), data.end());
    std::reverse(data.begin(), data.end());
    for (int i = 0; i < L; i++) {
        //std::cout << data[i].first << ", " << data[i].second << std::endl;
        SVs(i) = data[i].first;
    }
    logger::log->info("Done.");
    logger::log->info("Constructing sort matrix transforms...");
    PMatrix sort = PMatrix::Zero(L, L);
    Eigen::VectorXi sort_U = Eigen::VectorXi::LinSpaced(N, 0, N-1);
    Eigen::VectorXi sort_V = Eigen::VectorXi::LinSpaced(M, 0, M-1);
    for (int i = 0; i < L; i++) {
        sort(data[i].second, i) = 1;
        sort_U(i) = data[i].second;
        sort_V(i) = data[i].second;
    }
    Eigen::PermutationMatrix<Eigen::Dynamic> U_s;
    Eigen::PermutationMatrix<Eigen::Dynamic> V_s;
    U_s.indices() = sort_U;
    V_s.indices() = sort_V;
    logger::log->info("Done.");
    logger::log->info("Finalizing SVD...");
    PMatrix sigma = U_s.transpose()*sigma_prime*V_s;
    PMatrix U = U0*U_c*U_p*U_s;
    PMatrix V = V0*V_c*V_p*V_s;
    Grid grid = { .SVs = SVs.cast<Scalar>(), .U = U.cast<Scalar>(), .V = V.cast<Scalar>(),
                 .taus = taus.cast<Scalar>(), .omegas = omegas_d, .domegas = domegas,
                 .n_taus = n_taus, .n_omegas = static_cast<int>(omegas_d.size()), .beta = beta_d, .kernel = A.cast<Scalar>()};

    return grid;
}

spm::Grid spm::generate_grid(Vector omegas, Vector domegas, int n_taus, double beta, double recursion_tolerance, bool centrosymmetric) {
    assert(n_taus > 0);
    assert(beta > 0);
    logger::log->info("Creating grid...");
    logger::log->info("Centrosymmetric : {}", centrosymmetric);
    if (centrosymmetric) {
        return generate_centrosymmetric_grid(omegas, domegas, n_taus, beta, recursion_tolerance);
    }
    //Vector taus = symmetric_linspace(n_taus, static_cast<Scalar>(beta)/2, static_cast<Scalar>(beta)/2);
    Vector taus = Vector::LinSpaced(n_taus, 0, beta);
    int n_omegas = omegas.size();
    Matrix kernel = Matrix::Zero(n_taus, n_omegas);
    for (int it = 0; it < n_taus; it++) {
        for (int iw = 0 ; iw < n_omegas; iw++) {
            Scalar val = -domegas[iw]*std::exp(static_cast<Scalar>(beta)*omegas[iw] - taus[it]*omegas[iw])/(1 + std::exp(static_cast<Scalar>(beta)*omegas[iw]));
            //logger::log->info("Tau : {}, Omega : {}, K(tau, omega) : {}", taus[it], omegas[iw], val);
            kernel(it, iw) = val;
        }
    }
    PMatrix pkernel = kernel.cast<PScalar>();
    PSVD svd = recursive_svd_mp(pkernel, recursion_tolerance);
    logger::log->info("Performing direct SVD decomposition (recursion tolerance : {})...", recursion_tolerance);
        //svd = recursive_svd(kernel.cast<LScalar>(), recursion_tolerance);
    Grid grid { .SVs = svd.SVs.cast<Scalar>(), .U = svd.U.cast<Scalar>(), .V = svd.V.cast<Scalar>(),
                .taus = taus, .omegas = omegas, .domegas = domegas,
                .n_taus = n_taus, .n_omegas = n_omegas, .beta = beta,
                .kernel = kernel };
    logger::log->info("Done");
    //std::cout << grid.SVs << "\n\n";

    return grid;
}

spm::Vector spm::green_from_spectral(const Vector &spectral, Grid &grid) {
    return grid.U*grid.SVs.asDiagonal()*grid.V.transpose()*spectral;
}




void spm::run_spm(std::string settings_path) {
    auto [settings, green] = io::load_settings(settings_path);
    if (settings.debug.test_spectral) {
        logger::log->info("Spectral debug option enabled. Loading spectral function from file...");
        green = green_from_spectral(io::load_spectral(settings.debug.spectral_file), settings.grid);
        io::save_green(settings.output_path, green);
    }
    double l_min = std::log10(settings.admm_params.lambda_min);
    double l_max = std::log10(settings.admm_params.lambda_max);
    int N = std::round(l_max - l_min) + 1;
    N = std::round((settings.admm_params.lambda_res - 1)*(N - 1)) + N;
    Vector lambdas = Vector::LinSpaced(N, l_min, l_max);
    for (int i = 0 ; i < N; i++) {
        lambdas(i) = pow(10, lambdas(i));
    }
    double lambda_opt = 0;
    if (settings.debug.test_convergence) {
        Vector errors = admm_check_convergence(green, settings, settings.admm_params.lambda);
        io::save_vector(settings.output_path, errors, "errors");
    } else {
        if (settings.admm_params.override_lambda_opt) {
            lambda_opt = settings.admm_params.lambda;
            logger::log->info("Running single sim for lambda : {}", lambda_opt);
        } else {
            logger::log->info("Running {} lambda sims", N);
            Vector errors = calculate_lambda_errs(lambdas, green, settings);
            double a = (errors(errors.size() - 1) - errors(0))/(std::log(lambdas(lambdas.size() - 1)) - std::log(lambdas(0)));
            double div_max = std::numeric_limits<double>::min();
            for (int i = 0 ; i < errors.size(); i++) {
                double div = a*(std::log(lambdas(i)) - std::log(lambdas(0))) + errors(0);
                if (div / errors(i) > div_max) {
                    div_max = div / errors(i);
                    lambda_opt = lambdas(i);
                }
            }
            logger::log->info("Lambda opt: {}", lambda_opt);
            io::save_vector(settings.output_path, lambdas, "lambdas");
            io::save_vector(settings.output_path, errors, "errors");
        }


        Vector spectral = admm_minimize(green, settings, lambda_opt);
        Vector green_rc = settings.grid.kernel*spectral;
        io::save_spectral(settings.output_path, spectral);
        io::save_vector(settings.output_path, green_rc, "green_rc");
    }
}
