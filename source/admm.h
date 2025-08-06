#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <cassert>
#include "svd.h"
#include "structs.h"
#include "log.h"

namespace spm {
    Vector admm_check_convergence(const Vector &green, const SPM_settings &settings, double lambda);
    Vector admm_minimize(const Vector & green, const SPM_settings &settings);
    Vector admm_minimize(const Vector & green, const SPM_settings &settings, double lambda);
    Vector admm_minimize_raw(const Vector & y_prime, const Vector & SVs, const Vector & H_inv, const Vector & omega_prime, const Matrix & V, double alpha, double lambda, double mu, double mu_prime, double sum, bool fix_sum, int max_iter);
    Vector calculate_lambda_errs(const Vector & lambdas, const Vector & green, SPM_settings &settings);
    Vector soft_threshold(const Vector & input, double threshold);
    Vector positive_projection(const Vector & input);
}
