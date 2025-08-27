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

#include "admm.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

spm::Vector spm::positive_projection(const Vector &input) {
    Vector output(input.size());
    for (int i = 0; i < input.size(); ++i) {
        output(i) = input(i) * (input(i) > 0);
    }
    return output;
}

spm::Vector spm::soft_threshold(const Vector &input, double threshold) {
    Vector output(input.size());
    for (int i = 0; i < input.size(); ++i) {
        Scalar positive = input(i) - threshold;
        Scalar negative = -input(i) - threshold;
        output(i) = positive * (positive > 0) - negative * (negative > 0);
    }
    return output;
}

spm::Vector spm::admm_minimize(const Vector &green, const SPM_settings &settings) {
    return admm_minimize(green, settings, settings.admm_params.lambda);
}

spm::Vector spm::admm_check_convergence(const Vector &green, const SPM_settings &settings, double lambda_d) {
    auto & grid = settings.grid;
    auto V = grid.V;
    auto SVs = grid.SVs;
    auto U = grid.U;
    auto domegas = grid.domegas;

    int omega_dim = V.rows();
    int SV_dim = 0;
    auto lambda = static_cast<Scalar>(lambda_d);
    Scalar SV_tol = mpfr::abs(SVs(0))*settings.admm_params.sv_tol;
    for (int i = 0 ; i < SVs.size(); i++) {
        if (mpfr::abs(SVs(i)) < SV_tol) {
            break;
        }
        SV_dim++;
    }

    Scalar sum = -green(0) - green(green.size() - 1);

    Vector y_prime = U.transpose()*green;
    Vector omega_prime = V.transpose()*domegas;

    SVs.conservativeResize(SV_dim);
    U.conservativeResize(Eigen::NoChange, SV_dim);
    V.conservativeResize(Eigen::NoChange, SV_dim);
    y_prime.conservativeResize(SV_dim);
    omega_prime.conservativeResize(SV_dim);

    //std::cout << V.transpose()*V << std::endl;

    Vector x_prime = Vector::Zero(SV_dim);
    Vector z = Vector::Zero(omega_dim);
    Vector z_prime = Vector::Zero(SV_dim);
    Vector u = Vector::Zero(omega_dim);
    Vector u_prime = Vector::Zero(SV_dim);

    if (settings.admm_params.direct_inversion) {
        for (int i = 0; i < SV_dim; ++i) {
            x_prime(i) = y_prime(i)/SVs(i);
        }
        //std::cout << U*SVs.asDiagonal()*V.transpose()*V*x_prime << "\n";
        //std::cout << V*x_prime << "\n";
        //std::cout << U*SVs.asDiagonal()*V.transpose() << "\n";
        return V*x_prime;
    }

    Vector H_inv = Vector::Zero(SV_dim);
    Scalar alpha = 0;
    for (int i = 0 ; i < SV_dim; i++) {
        H_inv(i) = 1/(static_cast<Scalar>(settings.admm_params.mu + settings.admm_params.mu_prime) + (SVs(i) * SVs(i)));
        alpha += omega_prime(i)*omega_prime(i)*H_inv(i);
    }
    alpha = 1/alpha;
    Vector g = Vector::Zero(SV_dim);
    Vector temp = Vector::Zero(SV_dim);
    Vector x = Vector::Zero(omega_dim);
    Vector errors = Vector::Zero(settings.admm_params.max_iters);
    //Let's just try to do the simplest naive implementation
    for (int iter = 0 ; iter < settings.admm_params.max_iters ; iter++) {
        //First we do the x_prime update
        g = static_cast<Scalar>(settings.admm_params.mu_prime)*(u_prime - z_prime);
        g += static_cast<Scalar>(settings.admm_params.mu)*(V.transpose()*(u - z));
        g -= SVs.asDiagonal()*y_prime;
        Scalar omegaHinvg = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            omegaHinvg += omega_prime(i)*H_inv(i)*g(i);
        }
        //double chi_sqr = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            x_prime(i) = -H_inv(i)*g(i) + settings.admm_params.fix_sum*(alpha*H_inv(i)*omega_prime(i)*(sum + omegaHinvg));
            //chi_sqr += (SVs(i)*x_prime(i) - y_prime(i))*(SVs(i)*x_prime(i) - y_prime(i));
        }
        //logger::log->info("Chi_sqr : {}", chi_sqr);
        //x = V*x_prime;
        //double integral = x.dot(domegas);
        //logger::log->info("Spectral integral = {}", integral);
        //Not too bad! Then we move on to z_prime
        Scalar threshold = lambda/(static_cast<Scalar>(settings.admm_params.mu_prime) + (settings.admm_params.mu_prime == 0));
        for (int i = 0 ; i < SV_dim ; i++) {
            Scalar val = x_prime(i) + u_prime(i);
            Scalar positive = val - threshold;
            Scalar negative = -val - threshold;
            z_prime(i) = positive*(positive > 0) - negative*(negative > 0);
        }
        temp = V*x_prime + u;
        for (int i = 0 ; i < omega_dim ; i++) {
            z(i) = temp(i)*(temp(i) >= 0);
        }
        u_prime += x_prime - z_prime;
        u += V*x_prime - z;
        Scalar error = 0;
        for (int i = 0 ; i < SV_dim ; i++) {
            error += (y_prime(i) - SVs(i)*x_prime(i))*(y_prime(i) - SVs(i)*x_prime(i));
        }
        errors(iter) = error;
    }
    return errors;
}

spm::Vector spm::admm_minimize(const Vector &green, const SPM_settings &settings, double lambda_d) {
    auto & grid = settings.grid;
    auto V = grid.V;
    auto SVs = grid.SVs;
    auto U = grid.U;
    auto domegas = grid.domegas;


    int omega_dim = V.rows();
    int SV_dim = 0;
    Scalar SV_tol = SVs(0)*settings.admm_params.sv_tol;
    for (int i = 0 ; i < SVs.size(); i++) {
        if (SVs(i) < SV_tol) {
            break;
        }
        SV_dim++;
    }
    auto lambda = static_cast<Scalar>(lambda_d);
    Scalar sum = -green(0) - green(green.size() - 1);

    Vector y_prime = U.transpose()*green;
    Vector omega_prime = V.transpose()*domegas;

    SVs.conservativeResize(SV_dim);
    U.conservativeResize(Eigen::NoChange, SV_dim);
    V.conservativeResize(Eigen::NoChange, SV_dim);
    y_prime.conservativeResize(SV_dim);
    omega_prime.conservativeResize(SV_dim);

    //std::cout << V.transpose()*V << std::endl;

    Vector x_prime = Vector::Zero(SV_dim);
    Vector z = Vector::Zero(omega_dim);
    Vector z_prime = Vector::Zero(SV_dim);
    Vector u = Vector::Zero(omega_dim);
    Vector u_prime = Vector::Zero(SV_dim);

    if (settings.admm_params.direct_inversion) {
        for (int i = 0; i < SV_dim; ++i) {
            x_prime(i) = y_prime(i)/SVs(i);
        }
        //std::cout << U*SVs.asDiagonal()*V.transpose()*V*x_prime << "\n";
        //std::cout << V*x_prime << "\n";
        //std::cout << U*SVs.asDiagonal()*V.transpose() << "\n";
        return V*x_prime;
    }

    Vector H_inv = Vector::Zero(SV_dim);
    Scalar alpha = 0;
    for (int i = 0 ; i < SV_dim; i++) {
        H_inv(i) = 1/(static_cast<Scalar>(settings.admm_params.mu + settings.admm_params.mu_prime) + (SVs(i) * SVs(i)));
        alpha += omega_prime(i)*omega_prime(i)*H_inv(i);
    }
    alpha = 1/alpha;
    Vector g = Vector::Zero(SV_dim);
    Vector temp = Vector::Zero(SV_dim);
    Vector x = Vector::Zero(omega_dim);
    int fix_sum = settings.admm_params.fix_sum;
    //logger::log->info("FS : {}", fix_sum);
    //logger::log->info("Sum : {}", static_cast<double>(sum));
    //Let's just try to do the simplest naive implementation
    for (int iter = 0 ; iter < settings.admm_params.max_iters ; iter++) {
        //First we do the x_prime update
        g = static_cast<Scalar>(settings.admm_params.mu_prime)*(u_prime - z_prime);
        g += static_cast<Scalar>(settings.admm_params.mu)*(V.transpose()*(u - z));
        g -= SVs.asDiagonal()*y_prime;
        Scalar omegaHinvg = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            omegaHinvg += omega_prime(i)*H_inv(i)*g(i);
        }
        //double chi_sqr = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            x_prime(i) = -H_inv(i)*g(i) + (alpha*H_inv(i)*omega_prime(i)*(sum + omegaHinvg));
            //chi_sqr += (SVs(i)*x_prime(i) - y_prime(i))*(SVs(i)*x_prime(i) - y_prime(i));
        }
        //logger::log->info("Chi_sqr : {}", chi_sqr);
        //x = V*x_prime;
        //double integral = x.dot(domegas);
        //logger::log->info("Spectral integral = {}", integral);
        //Not too bad! Then we move on to z_prime
        Scalar threshold = lambda/(static_cast<Scalar>(settings.admm_params.mu_prime) + (settings.admm_params.mu_prime == 0));
        for (int i = 0 ; i < SV_dim ; i++) {
            Scalar val = x_prime(i) + u_prime(i);
            Scalar positive = val - threshold;
            Scalar negative = -val - threshold;
            z_prime(i) = positive*(positive > 0) - negative*(negative > 0);
        }
        temp = V*x_prime + u;
        for (int i = 0 ; i < omega_dim ; i++) {
            z(i) = temp(i)*(temp(i) >= 0);
        }
        u_prime += x_prime - z_prime;
        u += V*x_prime - z;
        //std::cout << "x_prime:\n";
        //logger::log->info("Sum : {}", static_cast<double>(omega_prime.dot(x_prime)));
        //Vector g_rec = U*SVs.asDiagonal()*x_prime;
        //logger::log->info("Grec start : {}, end : {}, sum : {}", -static_cast<double>(g_rec(0)), -static_cast<double>(g_rec(U.rows()-1)), -static_cast<double>(g_rec(U.rows()-1)+g_rec(0)));
    }
    return x_prime;
}

spm::Vector spm::admm_minimize_mp(const Vector &green, const PGrid & grid, const SPM_settings &settings) {
    return admm_minimize_mp(green, grid, settings, static_cast<Scalar>(settings.admm_params.lambda));
}

spm::Vector spm::admm_minimize_mp(const Vector &green, const PGrid & grid, const SPM_settings &settings, Scalar lambda) {
    auto V = grid.V;
    auto SVs = grid.SVs;
    auto U = grid.U;
    auto domegas = grid.domegas;

    int omega_dim = V.rows();
    int SV_dim = 0;
    Scalar SV_tol = SVs(0)*static_cast<Scalar>(settings.admm_params.sv_tol);
    for(int i = 0; i < SVs.size(); i++) {
        if(mpfr::abs(SVs(i)) < SV_tol) { break; }
        SV_dim++;
    }

    const auto sum = -green(0) - green(green.size() - 1);

    Vector y_prime = U.transpose()*green;
    Vector omega_prime = V.transpose()*domegas;

    SVs.conservativeResize(SV_dim);
    U.conservativeResize(Eigen::NoChange, SV_dim);
    V.conservativeResize(Eigen::NoChange, SV_dim);
    y_prime.conservativeResize(SV_dim);
    omega_prime.conservativeResize(SV_dim);

    //std::cout << V.transpose()*V << std::endl;

    Vector x_prime = Vector::Zero(SV_dim);
    Vector z = Vector::Zero(omega_dim);
    Vector z_prime = Vector::Zero(SV_dim);
    Vector u = Vector::Zero(omega_dim);
    Vector u_prime = Vector::Zero(SV_dim);

    if (settings.admm_params.direct_inversion) {
        for (int i = 0; i < SV_dim; ++i) {
            x_prime(i) = y_prime(i)/SVs(i);
        }
        //std::cout << U*SVs.asDiagonal()*V.transpose()*V*x_prime << "\n";
        //std::cout << V*x_prime << "\n";
        //std::cout << U*SVs.asDiagonal()*V.transpose() << "\n";
        Vector green_direct = V*x_prime;
        return green_direct.cast<Scalar>();
    }

    Vector H_inv = Vector::Zero(SV_dim);
    Scalar alpha = 0;
    for (int i = 0 ; i < SV_dim; i++) {
        H_inv(i) = 1/(settings.admm_params.mu + settings.admm_params.mu_prime + (SVs(i) * SVs(i)));
        alpha += omega_prime(i)*omega_prime(i)*H_inv(i);
    }
    alpha = 1/alpha;
    Vector g = Vector::Zero(SV_dim);
    Vector temp = Vector::Zero(SV_dim);
    Vector x = Vector::Zero(omega_dim);
    //Let's just try to do the simplest naive implementation
    for (int iter = 0 ; iter < settings.admm_params.max_iters ; iter++) {
        //First we do the x_prime update
        g = settings.admm_params.mu_prime*(u_prime - z_prime);
        g += settings.admm_params.mu*(V.transpose()*(u - z));
        g -= SVs.asDiagonal()*y_prime;
        Scalar omegaHinvg = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            omegaHinvg += omega_prime(i)*H_inv(i)*g(i);
        }
        //double chi_sqr = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            x_prime(i) = -H_inv(i)*g(i) + settings.admm_params.fix_sum*(alpha*H_inv(i)*omega_prime(i)*(sum + omegaHinvg));
            //chi_sqr += (SVs(i)*x_prime(i) - y_prime(i))*(SVs(i)*x_prime(i) - y_prime(i));
        }
        //logger::log->info("Chi_sqr : {}", chi_sqr);
        //x = V*x_prime;
        //double integral = x.dot(domegas);
        //logger::log->info("Spectral integral = {}", integral);
        //Not too bad! Then we move on to z_prime
        Scalar threshold = lambda/(settings.admm_params.mu_prime + (settings.admm_params.mu_prime == 0));
        for (int i = 0 ; i < SV_dim ; i++) {
            Scalar val = x_prime(i) + u_prime(i);
            Scalar positive = val - threshold;
            Scalar negative = -val - threshold;
            z_prime(i) = positive*(positive > 0) - negative*(negative > 0);
        }
        temp = V*x_prime + u;
        for (int i = 0 ; i < omega_dim ; i++) {
            z(i) = temp(i)*(temp(i) >= 0);
        }
        u_prime += x_prime - z_prime;
        u += V*x_prime - z;
    }
    Vector green_return = V*x_prime;
    return green_return.cast<Scalar>();
}

spm::Vector spm::admm_minimize_raw(const Vector & y_prime, const Vector & SVs, const Vector & H_inv, const Vector & omega_prime, const Matrix & V,
                                    Scalar alpha, Scalar lambda, Scalar mu, Scalar mu_prime, Scalar sum, bool fix_sum, int max_iter) {
    auto SV_dim = SVs.size();
    auto omega_dim = V.rows();
    Vector x_prime = Vector::Zero(SV_dim);
    Vector z = Vector::Zero(omega_dim);
    Vector z_prime = Vector::Zero(SV_dim);
    Vector u = Vector::Zero(omega_dim);
    Vector u_prime = Vector::Zero(SV_dim);
    Vector g = Vector::Zero(SV_dim);
    Vector temp = Vector::Zero(SV_dim);
    //Let's just try to do the simplest naive implementation
    for (int iter = 0 ; iter < max_iter ; iter++) {
        //First we do the x_prime update
        g = mu_prime*(u_prime - z_prime);
        g += mu*(V.transpose()*(u - z));
        g -= SVs.asDiagonal()*y_prime;
        Scalar omegaHinvg = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            omegaHinvg += omega_prime(i)*H_inv(i)*g(i);
        }
        //double chi_sqr = 0;
        for (int i = 0 ; i < SV_dim; i++) {
            x_prime(i) = -H_inv(i)*g(i) + fix_sum*(alpha*H_inv(i)*omega_prime(i)*(sum + omegaHinvg));
            //chi_sqr += (SVs(i)*x_prime(i) - y_prime(i))*(SVs(i)*x_prime(i) - y_prime(i));
        }
        //logger::log->info("Chi_sqr : {}", chi_sqr);
        //x = V*x_prime;
        //double integral = x.dot(domegas);
        //logger::log->info("Spectral integral = {}", integral);
        //Not too bad! Then we move on to z_prime
        Scalar threshold = lambda/(mu_prime + (mu_prime == 0));
        for (int i = 0 ; i < SV_dim ; i++) {
            Scalar val = x_prime(i) + u_prime(i);
            Scalar positive = val - threshold;
            Scalar negative = -val - threshold;
            z_prime(i) = positive*(positive > 0) - negative*(negative > 0);
        }
        temp = V*x_prime + u;
        for (int i = 0 ; i < omega_dim ; i++) {
            z(i) = temp(i)*(temp(i) >= 0);
        }
        u_prime += x_prime - z_prime;
        u += V*x_prime - z;
    }
    return x_prime;
}

spm::Vector spm::calculate_lambda_errs(const Vector &lambdas, const Vector & green, SPM_settings &settings) {
    Vector chi_sqr = Vector::Zero(lambdas.size());

    auto & grid = settings.grid;
    auto V = grid.V;
    auto SVs = grid.SVs;
    auto U = grid.U;
    auto domegas = grid.domegas;

    int omega_dim = V.rows();
    int SV_dim = 1;
    Scalar SV_tol = SVs(0)*static_cast<Scalar>(settings.admm_params.sv_tol);
    for (int i = 1 ; i < SVs.size(); i++) {
        if (SVs(i) < SV_tol) {
            break;
        }
        SV_dim++;
    }

    Scalar sum = -green(0) - green(green.size() - 1);

    Vector y_prime = U.transpose()*green;
    Vector omega_prime = V.transpose()*domegas;

    SVs.conservativeResize(SV_dim);
    U.conservativeResize(Eigen::NoChange, SV_dim);
    V.conservativeResize(Eigen::NoChange, SV_dim);
    y_prime.conservativeResize(SV_dim);
    omega_prime.conservativeResize(SV_dim);

    Vector H_inv = Vector::Zero(SV_dim);
    Scalar alpha = 0;
    for (int i = 0 ; i < SV_dim; i++) {
        H_inv(i) = 1/(settings.admm_params.mu + settings.admm_params.mu_prime + (SVs(i) * SVs(i)));
        alpha += omega_prime(i)*omega_prime(i)*H_inv(i);
    }
    alpha = 1/alpha;


    for (int i = 0 ; i < lambdas.size() ; i++) {
        logger::log->info("Calculating lambda errs for {}", static_cast<double>(lambdas(i)));
        Vector rho_prime = admm_minimize_raw(y_prime, SVs, H_inv, omega_prime, V, alpha, lambdas(i),
                                static_cast<Scalar>(settings.admm_params.mu), static_cast<Scalar>(settings.admm_params.mu_prime),
                                sum, settings.admm_params.fix_sum, settings.admm_params.max_iters);
        auto & kernel = settings.grid.kernel;
        Vector green_rc = U*SVs.asDiagonal()*rho_prime;
        //logger::log->info("kernel dims : {}, {}", kernel.rows(), kernel.cols());
        //logger::log->info("green_rc dims : {}, {}", green_rc.rows(), green_rc.cols());
        //logger::log->info("green dims : {}, {}", greens.rows(), greens.cols());
        chi_sqr(i) = (green - U*SVs.asDiagonal()*rho_prime).squaredNorm()/2;
        logger::log->info("Chi_sqr : {}", static_cast<double>(chi_sqr(i)));
    }
    return chi_sqr;
}
