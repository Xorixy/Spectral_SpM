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
#include "svd.h"

namespace spm {
    Grid generate_centrosymmetric_grid(Vector omegas_d, Vector domegas, int n_taus, double beta_d, double recursion_tolerance = -1.0);
    Grid generate_grid(Vector omegas, Vector domegas, int n_taus, double beta, double recursion_tolerance = -1.0, bool centrosymmetric = false);
    Vector green_from_spectral(const Vector & spectral, Grid & grid);
    void run_spm(std::string settings_path);
}
