#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>
#include <cassert>
#include "svd.h"
#include "structs.h"

namespace spm {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;



    Vector admm_minimize(Vector & green, Grid grid, ADMM_params params);
    Vector soft_threshold(Vector & input, double threshold);
    Vector positive_projection(Vector & input);
}
