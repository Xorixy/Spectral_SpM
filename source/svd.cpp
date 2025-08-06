#include "svd.h"

spm::SVD spm::recursive_svd(const Eigen::MatrixXd &A, double tol) {
    if (tol > 1.0) {
        throw std::invalid_argument("tolerance must be less than 1.0 for recursive_svd");
    }
    auto svd = Eigen::JacobiSVD(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    int l = svd.singularValues().size();
    logger::log->info("SVD decomposition done. Number of SVs : {}", svd.singularValues().size());
    int n = svd.matrixU().cols();
    int m = svd.matrixV().cols();
    auto lambda_max = std::abs(svd.singularValues()(0));
    int p = -1;
    for (int i = 1; i < svd.singularValues().size(); ++i) {
        if (std::abs(svd.singularValues()(i)) < std::abs(lambda_max)*tol) {
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

