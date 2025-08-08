#include "svd.h"

spm::LMatrix spm::get_j_matrix(int n) {
    LMatrix J = LMatrix::Zero(n, n);
    for (int i = 0; i < n; i++) {
        J(n - i - 1, i) = static_cast<Scalar>(1);
    }
    return J;
}

spm::Vector spm::symmetric_linspace(int n, Scalar max, Scalar offset) {
    if (n <= 1) {
        throw std::invalid_argument("spm::symmetric_linspace: n must be > 1");
    }
    max = std::abs(max);
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



void spm::test_centrosymmetric() {
    double tol = -1;
    Scalar beta = 1.32;
    Scalar omega_max = 1.56;
    Scalar sqrt2 = std::sqrt(static_cast<Scalar>(2));
    Scalar sqrt2inv = 1/sqrt2;
    int N = 21;
    int M = 7;
    int p = N/2;
    int a = N % 2;
    int q = M/2;
    int b = M % 2;
    LVector taus = symmetric_linspace(N, beta/2, beta/2);
    LVector omegas = symmetric_linspace(M, omega_max);
    LMatrix A = LMatrix::Zero(N, M);
    for (int i = 0; i < N; i++) {
        for (int j = 0 ; j < M ; j++) {
            A(i, j) = -std::exp(-taus(i)*omegas(j))/(1 + std::exp(-beta*omegas(j)));
        }
    }
    LMatrix Ip = LMatrix::Identity(p, p);
    LMatrix Iq = LMatrix::Identity(q, q);
    LMatrix Jp = get_j_matrix(p);
    LMatrix Jq = get_j_matrix(q);
    LMatrix U0 = LMatrix::Zero(N, N);
    U0.block(0, 0, p, p) = Ip;
    U0.block(0, p+a, p, p) = Ip;
    U0.block(p+a, 0, p, p) = Jp;
    U0.block(p+a, p+a, p, p) = -Jp;
    if (a == 1) {
        U0(p+a-1, p+a-1) = sqrt2;
    }
    U0 *= 1/std::sqrt(2);
    LMatrix V0 = LMatrix::Zero(M, M);
    V0.block(0, 0, q, q) = Iq;
    V0.block(0, q+b, q, q) = Iq;
    V0.block(q+b,0,q,q) = Jq;
    V0.block(q+b,q+b,q,q) = -Jq;
    if (b == 1) {
        V0(q+b-1, q+b-1) = std::sqrt(2);
    }
    V0 *= 1/std::sqrt(2);
    LMatrix A_block = A;
    A_block = U0.transpose()*A_block;
    A_block = A_block*V0;
    //std::cout << A << "\n\n";
    //std::cout << U0 << "\n\n";
    //std::cout << V0 << "\n\n";
    //std::cout << A_block << "\n\n";
    LMatrix B1 = LMatrix::Zero(p+a, q+b);
    LMatrix B2 = LMatrix::Zero(p, q);
    for (int i = 0 ; i < p ; i++) {
        for (int j = 0 ; j < q ; j++) {
            B1(i,j) = std::sinh(taus(i)*omegas(j))*std::tanh(omegas(j)*beta/2) - std::cosh(taus(i)*omegas(j));
            B2(i, j) = std::sinh(taus(i)*omegas(j)) - std::cosh(taus(i)*omegas(j))*std::tanh(omegas(j)*beta/2);
        }
    }
    if (a == 1) {
        for (int j = 0 ; j < q ; j++) {
            B1(p+a-1, j) = -1/(sqrt2*std::cosh(omegas(j)*beta/2));
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
    LMatrix B = LMatrix::Zero(N, M);
    B.block(0, 0, p+a, q+b) = B1;
    B.block(p+a, q+b, p, q) = B2;
    //std::cout << A << "\n\n";
    //std::cout << U0*B*V0.transpose() << "\n\n";
    //std::cout << B1 << "\n\n";
    //std::cout << B2 << "\n\n";
    auto svd_1 = recursive_svd(B1, tol);
    auto svd_2 = recursive_svd(B2, tol);
    LMatrix sigma_tilde = LMatrix::Zero(N, M);
    //std::cout << sigma_tilde << "\n\n";
    //std::cout << svd_1.SVs.asDiagonal().toDenseMatrix() << "\n\n";
    //std::cout << svd_2.SVs.asDiagonal().toDenseMatrix() << "\n\n";
    int l = std::min(p, q);
    int lc = std::min(p+a, q+b);
    //fmt::print("l : {}, lc : {}\n", l, lc);
    sigma_tilde.block(0, 0, lc, lc) = svd_1.SVs.asDiagonal().toDenseMatrix();
    sigma_tilde.block(p+a, q+b, l, l) = svd_2.SVs.asDiagonal().toDenseMatrix();
    //std::cout << sigma_tilde << "\n\n";
    LMatrix U_c = LMatrix::Zero(N, N);
    U_c.block(0, 0, p+a, p+a) = svd_1.U;
    U_c.block(p+a, p+a, p, p) = svd_2.U;
    LMatrix V_c = LMatrix::Zero(M, M);
    V_c.block(0, 0, q+b, q+b) = svd_1.V;
    V_c.block(q+b, q+b, q, q) = svd_2.V;
    //std::cout << U0*U_c*sigma_tilde*V_c.transpose()*V0.transpose() << "\n\n";
    //std::cout << A << "\n\n";
    int gap = p + a - q - b;
    LMatrix U_p = LMatrix::Identity(N, N);
    LMatrix V_p = LMatrix::Identity(M, M);
    if (gap > 0) {
        int perm_dim = gap + p;
        LMatrix perm = LMatrix::Zero(perm_dim, perm_dim);
        for (int i = 0; i < perm_dim; i++) {
            perm( i, (perm_dim + i - gap) % perm_dim) = 1;
        }
        U_p.block(N - perm_dim, N - perm_dim, perm_dim, perm_dim) = perm;
    }
    if (gap < 0) {
        gap = -gap;
        int perm_dim = gap + q;
        LMatrix perm = LMatrix::Zero(perm_dim, perm_dim);
        for (int j = 0; j < perm_dim; j++) {
            perm(j,(perm_dim + j - gap) % perm_dim) = 1;
        }
        V_p.block(M - perm_dim, M - perm_dim, perm_dim, perm_dim) = perm;

    }
    LMatrix sigma_prime = U_p.transpose()*sigma_tilde*V_p;
    //std::cout << sigma_prime << "\n\n";
    int L = std::min(N, M);
    std::vector<std::pair<Scalar, int>> data(L);
    for (int i = 0; i < L; i++) {
        data[i] = {sigma_prime(i, i), i};
        //std::cout << data[i].first << ", " << data[i].second << std::endl;
    }
    std::ranges::sort(data.begin(), data.end());
    std::reverse(data.begin(), data.end());
    for (int i = 0; i < L; i++) {
        //std::cout << data[i].first << ", " << data[i].second << std::endl;
    }
    LMatrix sort = LMatrix::Zero(L, L);
    for (int i = 0; i < L; i++) {
        sort(data[i].second, i) = 1;
    }
    LMatrix U_s = LMatrix::Identity(N, N);
    U_s.block(0, 0, L, L) = sort;
    LMatrix V_s = LMatrix::Identity(M, M);
    V_s.block(0, 0, L, L) = sort;
    //std::cout << sort << "\n\n";
    //std::cout << U_s << "\n\n";
    //std::cout << V_s << "\n\n";
    LMatrix sigma = U_s.transpose()*sigma_prime*V_s;
    //std::cout << sigma << std::endl;
    LMatrix U = U0*U_c*U_p*U_s;
    LMatrix V = V0*V_c*V_p*V_s;
    //std::cout << (A - U*sigma*V.transpose());
    std::cout << std::numeric_limits<long double>::digits10 << std::endl;
}




spm::SVD spm::centrosymmetric_matrix_svd(const LMatrix &A, double tol) {
    logger::log->info("Splitting kernel into centrosymmetric components...");
    int N = A.rows();
    int M = A.cols();
    int p = N/2;
    int a = N % 2;
    int q = M / 2;
    int b = M % 2;
    LMatrix A1 = A.block(0, 0, p, q);
    LMatrix A2 = LMatrix::Zero(p, q);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            A2(i, j) = A(2*p + a - 1 - i, j);
        }
    }
    LMatrix B1 = LMatrix::Zero(p + a, q + b);
    LMatrix B2 = LMatrix::Zero(p, q);
    B1.block(0, 0, p, q) = A1 + A2;
    B2 = A1 - A2;
    if (a == 1) {
        for (int j = 0 ; j < q ; j++) {
            B1(p + a - 1, j) = std::sqrt(static_cast<Scalar>(2))*A(p + a - 1, j);
        }
    }
    if (b == 1) {
        for (int i = 0 ; i < q ; i++) {
            B1(i, q + b - 1) = std::sqrt(static_cast<Scalar>(2))*A(i, q + b - 1);
        }
    }
    if (a == 1 && b == 1) {
        B1(p+a-1, q+b-1) = A(p+a-1, q+b-1);
    }
}


spm::SVD spm::recursive_svd(const LMatrix & A, double tol) {
    if (tol > 1.0) {
        throw std::invalid_argument("tolerance must be less than 1.0 for recursive_svd");
    }

    auto svd = Eigen::JacobiSVD(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
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
    LMatrix S_prime = svd.matrixU().transpose()*A*svd.matrixV();
    logger::log->info("Matrix S' size : {}, {}", S_prime.rows(), S_prime.cols());
    LMatrix X = S_prime.block(p, p, (l - p), (l - p));
    logger::log->info("Matrix X size : {}, {}", X.rows(), X.cols());
    SVD svd_X = recursive_svd(X, tol);
    LVector SVs = svd.singularValues();
    LMatrix U_prime = LMatrix::Identity(n, n);
    U_prime.block(p, p, l-p, l-p) = svd_X.U;
    LMatrix V_prime = LMatrix::Identity(m, m);
    V_prime.block(p, p, l-p, l-p) = svd_X.V;
    SVs.tail(l-p) = svd_X.SVs;
    LMatrix U = svd.matrixU()*U_prime;
    LMatrix V = svd.matrixV()*V_prime;
    return {SVs, U, V};
}

