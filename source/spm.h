#pragma once
#include "svd.h"
#include "admm.h"
#include <Eigen/Core>
#include <cmath>
#include "../log.h"
#include <h5pp/h5pp.h>
#include "structs.h"

namespace spm {

    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;

    Grid generate_grid(Vector omegas, Vector domegas, int n_taus, double beta);

    void save_grid(Grid & grid, std::string filename);
    Grid load_grid(const std::string filename);



    void test_spm();
}
