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

        Matrix kernel;
    };
    struct ADMM_params {
        double lambda { 1 };
        double lambda_max { 1 };
        double lambda_min { 10e-10 };
        double lambda_res { 5 };
        double sv_tol { 10e-10 };
        double mu { 1 };
        double mu_prime { 1 };
        int max_iters { 10000 };
        bool fix_sum { true };
        bool direct_inversion { false };
        bool override_lambda_opt { false };
    };
    struct DebugSettings {
        bool test_spectral { false };
        std::string spectral_file;
    };
    struct SPM_settings {
        Grid grid;
        ADMM_params admm_params;
        DebugSettings debug;
        std::string output_path;
    };
}