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
    Vector admm_minimize_mp(const Vector & green, const PGrid &grid, const SPM_settings &settings);
    Vector admm_minimize_mp(const Vector & green, const PGrid &grid, const SPM_settings &settings, Scalar lambda);
    Vector admm_minimize_raw(const Vector & y_prime, const Vector & SVs, const Vector & H_inv, const Vector & omega_prime, const Matrix & V,
                             Scalar alpha, Scalar lambda, Scalar mu, Scalar mu_prime, Scalar sum, bool fix_sum, int max_iter);
    Vector calculate_lambda_errs(const Vector & lambdas, const Vector & green, SPM_settings &settings);
    Vector soft_threshold(const Vector & input, double threshold);
    Vector positive_projection(const Vector & input);
}
