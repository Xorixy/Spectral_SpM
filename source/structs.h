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
#include <optional>
#include "mpreal.h"
#include <unsupported/Eigen/MPRealSupport>


namespace spm {
    using Scalar = double;
    using Matrix = Eigen::Matrix<Scalar, -1, -1>;
    using Vector = Eigen::Vector<Scalar, -1>;
    using PScalar = mpfr::mpreal;
    using PMatrix = Eigen::Matrix<PScalar, -1, -1>;
    using PVector = Eigen::Vector<PScalar, -1>;

    struct CSMatrix {
        //This struct contains all the information required to store a centrosymmetric matrix,
        //i.e. an N x M matrix A which satisfies A_{(N - i + 1), (M - j + 1)} = A_{i,j}
        //Letting J_k be the k x k matrix with ones on the second diagonal and zeros elsewhere, this condition can be equivalently stated as
        //J_N A J_M = A.
        //There are four cases of this matrix corresponding to all combinations of N/M being even/odd.
        //Letting p = N/2 rounded down and q = M/2 rounded down we have:
        //N odd M odd:
        //A = | A_1         u           A_2 J_q     |
        //    | v^T         alpha       v^T J_q     |
        //    | J_p A_2     J_p u       J_p A_1 J_q |
        //
        //N even M odd:
        //A = | A_1         u           A_2 J_q     |
        //    | J_p A_2     J_p u       J_p A_1 J_q |
        //
        //N odd M even:
        //A = | A_1         A_2 J_q     |
        //    | v^T         v^T J_q     |
        //    | J_p A_2     J_p A_1 J_q |
        //
        //N even M even:
        //A = | A_1         A_2 J_q     |
        //    | J_p A_2     J_p A_1 J_q |
        //
        //where A_{1,2} are p x q matrices, u is a p vector, v is a q vector and alpha is a scalar.
        //Note that all combinations contain A_{1,2} and only odd ones contain u/v/alpha.

        Matrix A_1;
        Matrix A_2;
        std::optional<Vector> u;
        std::optional<Vector> v;
        std::optional<Scalar> alpha;
    };

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

    struct PGrid {
        PVector SVs;
        PMatrix U;
        PMatrix V;

        PVector taus;
        PVector omegas;
        PVector domegas;

        int n_taus;
        int n_omegas;
        PScalar beta;

        PMatrix kernel;
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
        bool test_convergence { false };
        bool test_spectral { false };
        std::string spectral_file;
    };
    struct SPM_settings {
        Grid grid;
        ADMM_params admm_params;
        DebugSettings debug;
        std::string output_path;
        int precision { 100 };
        bool centrosymmetric { false };
    };
    struct PSVD {
        PVector SVs;
        PMatrix U;
        PMatrix V;
    };
    struct SVD {
        Vector SVs;
        Matrix U;
        Matrix V;
    };
}