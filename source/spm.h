#pragma once
#include "svd.h"
#include "admm.h"
#include <Eigen/Core>
#include <cmath>
#include "log.h"
#include <fstream>
#include <h5pp/h5pp.h>
#include "structs.h"
#include <nlohmann/json.hpp>
#include <vector>
#include "io.h"

namespace spm {
    Grid generate_grid(Vector omegas, Vector domegas, int n_taus, double beta);
    Vector green_from_spectral(const Vector & spectral, Grid & grid);
    void run_spm(std::string settings_path);
}
