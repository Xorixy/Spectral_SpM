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

#include "svd.h"

spm::Matrix spm::get_j_matrix(int n) {
    Matrix J = Matrix::Zero(n, n);
    for (int i = 0; i < n; i++) {
        J(n - i - 1, i) = static_cast<Scalar>(1);
    }
    return J;
}

spm::Vector spm::symmetric_linspace(int n, Scalar max, Scalar offset) {
    if (n <= 1) {
        throw std::invalid_argument("spm::symmetric_linspace: n must be > 1");
    }
    max = mpfr::abs(max);
    int k = n/2;
    Vector v = Vector::Zero(n);
    for (int i = 0; i < k; i++) {
        v(i) = max*(static_cast<Scalar>(2*i)/(static_cast<Scalar>(n - 1)) - 1);
        v(n - i - 1) = -v(i);
        v(i) += offset;
        v(n - i - 1) += offset;
    }
    if (n % 2 == 1) {
        v(k) = offset;
    }
    return v;
}




spm::Vector spm::test_centrosymmetric() {
    /*
    Eigen::Matrix3d MX;
    MX << 1,2,3,4,5,6,7,8,9;
    std::vector<int> vector_of_indices = {1,2,0};
    Eigen::VectorXi indices(MX.cols());
    for(long i = 0; i < indices.size(); ++i) {
        indices[i] = vector_of_indices[i];
    }
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PM;
    PM.indices() = indices;
    std::cout << MX << "\n\n";
    std::cout << PM.toDenseMatrix() << "\n\n";
    std::cout << PM * MX << "\n\n";
    std::cout << MX * PM << "\n\n";
    return;
    */
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(100));
    const auto start{std::chrono::steady_clock::now()};
    double tol = -1;
    Scalar beta = 1.32;
    Scalar omega_max = 1.56;
    Scalar sqrt2 = mpfr::sqrt(static_cast<Scalar>(2));
    Scalar sqrt2inv = static_cast<Scalar>(1)/sqrt2;
    int N = 101;
    int M = 101;
    int p = N/2;
    int a = N % 2;
    int q = M/2;
    int b = M % 2;
    logger::log->info("Generating block kernel...");
    Vector taus = symmetric_linspace(N, beta/2, beta/2);
    Vector omegas = symmetric_linspace(M, omega_max);
    Matrix A = Matrix::Zero(N, M);
    for (int i = 0; i < N; i++) {
        for (int j = 0 ; j < M ; j++) {
            A(i, j) = -mpfr::exp(-taus(i)*omegas(j))/(1 + mpfr::exp(-beta*omegas(j)));
        }
    }
    Matrix Ip = Matrix::Identity(p, p);
    Matrix Iq = Matrix::Identity(q, q);
    Matrix Jp = get_j_matrix(p);
    Matrix Jq = get_j_matrix(q);
    Matrix U0 = Matrix::Zero(N, N);
    U0.block(0, 0, p, p) = Ip;
    U0.block(0, p+a, p, p) = Ip;
    U0.block(p+a, 0, p, p) = Jp;
    U0.block(p+a, p+a, p, p) = -Jp;
    if (a == 1) {
        U0(p+a-1, p+a-1) = sqrt2;
    }
    U0 *= sqrt2inv;
    Matrix V0 = Matrix::Zero(M, M);
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
    Matrix B1 = Matrix::Zero(p+a, q+b);
    Matrix B2 = Matrix::Zero(p, q);
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
        B1(p+a-1, q+b-1) = static_cast<Scalar>(-0.5);
    }
    Matrix B = Matrix::Zero(N, M);
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
    Matrix sigma_tilde = Matrix::Zero(N, M);
    //std::cout << sigma_tilde << "\n\n";
    //std::cout << svd_1.SVs.asDiagonal().toDenseMatrix() << "\n\n";
    //std::cout << svd_2.SVs.asDiagonal().toDenseMatrix() << "\n\n";
    int l = std::min(p, q);
    int lc = std::min(p+a, q+b);
    //fmt::print("l : {}, lc : {}\n", l, lc);
    sigma_tilde.block(0, 0, lc, lc) = svd_1.SVs.asDiagonal().toDenseMatrix();
    sigma_tilde.block(p+a, q+b, l, l) = svd_2.SVs.asDiagonal().toDenseMatrix();
    //std::cout << sigma_tilde << "\n\n";
    Matrix U_c = Matrix::Zero(N, N);
    U_c.block(0, 0, p+a, p+a) = svd_1.U;
    U_c.block(p+a, p+a, p, p) = svd_2.U;
    Matrix V_c = Matrix::Zero(M, M);
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
    Eigen::PermutationMatrix<Eigen::Dynamic> U_p2;
    Eigen::PermutationMatrix<Eigen::Dynamic> V_p2;
    Eigen::VectorXi perm_U = Eigen::VectorXi::LinSpaced(N, 0, N-1);
    Eigen::VectorXi perm_V = Eigen::VectorXi::LinSpaced(M, 0, M-1);
    Matrix U_p = Matrix::Identity(N, N);
    Matrix V_p = Matrix::Identity(M, M);
    if (gap > 0) {
        int perm_dim = gap + p;
        Matrix perm = Matrix::Zero(perm_dim, perm_dim);
        for (int i = 0; i < perm_dim; i++) {
            perm( i, (perm_dim + i - gap) % perm_dim) = 1;
            perm_U(q + b + (perm_dim + i - gap) % perm_dim) = i + q + b;
        }
        U_p.block(N - perm_dim, N - perm_dim, perm_dim, perm_dim) = perm;
    }
    if (gap < 0) {
        gap = -gap;
        int perm_dim = gap + q;
        Matrix perm = Matrix::Zero(perm_dim, perm_dim);
        for (int j = 0; j < perm_dim; j++) {
            perm(j,(perm_dim + j - gap) % perm_dim) = 1;
            perm_V(p + a + (perm_dim + j - gap) % perm_dim) = j + p + a;
        }
        V_p.block(M - perm_dim, M - perm_dim, perm_dim, perm_dim) = perm;
    }
    U_p2.indices() = perm_U;
    V_p2.indices() = perm_V;
    //std::cout << U_p << "\n\n";
    //std::cout << U_p2.toDenseMatrix() << "\n\n";
    //std::cout << V_p << "\n\n";
    //std::cout << V_p2.toDenseMatrix() << "\n\n";
    Matrix sigma_prime = U_p.transpose()*sigma_tilde*V_p;
    logger::log->info("Done.");
    logger::log->info("Sorting SVs...");
    //std::cout << sigma_prime << "\n\n";
    int L = std::min(N, M);
    std::vector<std::pair<Scalar, int>> data(L);
    Vector SVs = Vector::Zero(L);
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
    Matrix sort = Matrix::Zero(L, L);
    Eigen::VectorXi sort_U = Eigen::VectorXi::LinSpaced(N, 0, N-1);
    Eigen::VectorXi sort_V = Eigen::VectorXi::LinSpaced(M, 0, M-1);
    for (int i = 0; i < L; i++) {
        sort(data[i].second, i) = 1;
        sort_U(i) = data[i].second;
        sort_V(i) = data[i].second;
    }
    Matrix U_s = Matrix::Identity(N, N);
    U_s.block(0, 0, L, L) = sort;
    Matrix V_s = Matrix::Identity(M, M);
    V_s.block(0, 0, L, L) = sort;
    Eigen::PermutationMatrix<Eigen::Dynamic> U_s2;
    Eigen::PermutationMatrix<Eigen::Dynamic> V_s2;
    U_s2.indices() = sort_U;
    V_s2.indices() = sort_V;
    //std::cout << sort << "\n\n";
    //std::cout << U_s << "\n\n";
    //std::cout << U_s2.toDenseMatrix() << "\n\n";
    //std::cout << V_s << "\n\n";
    //std::cout << V_s2.toDenseMatrix() << "\n\n";
    logger::log->info("Done.");
    logger::log->info("Finalizing SVD...");
    Matrix sigma = U_s.transpose()*sigma_prime*V_s;
    //std::cout << sigma << std::endl;
    logger::log->info("Calculating U...");
    Matrix U = U0*U_c*U_p2*U_s2;
    logger::log->info("Calculating V...");
    Matrix V = V0*V_c*V_p2*V_s2;
    logger::log->info("Calculating sigma...");
    //std::cout << (A - U*sigma*V.transpose()) << "\n\n";
    for (int i = 0 ; i < L ; i++) {
        std::cout << SVs(i) << "\n";
    }

    const auto finish{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{finish - start};
    logger::log->info("Done.");
    logger::log->info("Elapsed : {}",elapsed_seconds.count());
    h5pp::File file("../../results/xtest.h5", h5pp::FileAccess::READWRITE);
    file.writeDataset(SVs, "SVs");
    return SVs;
}

spm::SVD spm::recursive_svd(const Matrix & A, double tol) {
    if (tol > 1.0) {
        throw std::invalid_argument("tolerance must be less than 1.0 for recursive_svd");
    }
    logger::log->info("Recursive SVD step for matrix with sizes {} X {}", A.rows(), A.cols());
    auto svd = Eigen::BDCSVD(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    int l = svd.singularValues().size();
    logger::log->info("SVD decomposition done. Number of SVs : {}", svd.singularValues().size());
    int n = svd.matrixU().cols();
    int m = svd.matrixV().cols();
    auto lambda_max = svd.singularValues()(0);
    int p = -1;
    for (int i = 1; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()(i) < lambda_max*tol) {
            p = i;
            break;
        }
    }
    if (p == -1) {
        logger::log->info("All SVs within tolerance, returning full SVD");
        return {svd.singularValues(), svd.matrixU(), svd.matrixV()};
    }
    logger::log->info("Singular values within tolerance: {}", p);
    logger::log->info("Running recursive step...");
    Matrix S_prime = svd.matrixU().transpose()*A*svd.matrixV();
    logger::log->info("Matrix S' size : {}, {}", S_prime.rows(), S_prime.cols());
    Matrix X = S_prime.block(p, p, (l - p), (l - p));
    logger::log->info("Matrix X size : {}, {}", X.rows(), X.cols());
    SVD svd_X = recursive_svd(X, tol);
    Vector SVs = svd.singularValues();
    Matrix U_prime = Matrix::Identity(n, n);
    U_prime.block(p, p, l-p, l-p) = svd_X.U;
    Matrix V_prime = Matrix::Identity(m, m);
    V_prime.block(p, p, l-p, l-p) = svd_X.V;
    SVs.tail(l-p) = svd_X.SVs;
    Matrix U = svd.matrixU()*U_prime;
    Matrix V = svd.matrixV()*V_prime;
    return {SVs, U, V};
}

spm::PSVD spm::recursive_svd_mp(const Matrix & A, double tol) {
    if (tol > 1.0) {
        throw std::invalid_argument("tolerance must be less than 1.0 for recursive_svd");
    }
    logger::log->info("Recursive SVD step for matrix with sizes {} X {}", A.rows(), A.cols());
    auto svd = Eigen::BDCSVD(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    int l = svd.singularValues().size();
    logger::log->info("SVD decomposition done. Number of SVs : {}", svd.singularValues().size());
    int n = svd.matrixU().cols();
    int m = svd.matrixV().cols();
    auto lambda_max = svd.singularValues()(0);
    int p = -1;
    for (int i = 1; i < svd.singularValues().size(); ++i) {
        if (svd.singularValues()(i) < lambda_max*tol) {
            p = i;
            break;
        }
    }
    if (p == -1) {
        logger::log->info("All SVs within tolerance, returning full SVD");
        return {svd.singularValues(), svd.matrixU(), svd.matrixV()};
    }
    logger::log->info("Singular values within tolerance: {}", p);
    logger::log->info("Running recursive step...");
    Matrix S_prime = svd.matrixU().transpose()*A*svd.matrixV();
    logger::log->info("Matrix S' size : {}, {}", S_prime.rows(), S_prime.cols());
    Matrix X = S_prime.block(p, p, (l - p), (l - p));
    logger::log->info("Matrix X size : {}, {}", X.rows(), X.cols());
    PSVD svd_X = recursive_svd_mp(X, tol);
    Vector SVs = svd.singularValues();
    Matrix U_prime = Matrix::Identity(n, n);
    U_prime.block(p, p, l-p, l-p) = svd_X.U;
    Matrix V_prime = Matrix::Identity(m, m);
    V_prime.block(p, p, l-p, l-p) = svd_X.V;
    SVs.tail(l-p) = svd_X.SVs;
    Matrix U = svd.matrixU()*U_prime;
    Matrix V = svd.matrixV()*V_prime;
    return {SVs, U, V};
}


