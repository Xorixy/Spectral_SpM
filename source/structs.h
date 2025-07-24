#pragma once
#include <Eigen/Core>

namespace spm {
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    struct Grid {
        Vector SVs;
        Matrix U;
        Matrix V;

        Vector taus;
        Vector omegas;
        Vector domegas;

        int n_taus;
        int n_omegas;
        double beta;
    };
    struct ADMM_params {
        double lambda { 1 };
        double sv_tol { 10e-10 };
        double mu { 1 };
        double mu_prime { 1 };
    };
}