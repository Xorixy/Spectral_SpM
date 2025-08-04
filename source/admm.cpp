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
        double positive = input(i) - threshold;
        double negative = -input(i) - threshold;
        output(i) = positive * (positive > 0) - negative * (negative > 0);
    }
    return output;
}

spm::Vector spm::admm_minimize(const Vector &green, const SPM_settings &settings) {
    return admm_minimize(green, settings, settings.admm_params.lambda);
}

spm::Vector spm::admm_minimize(const Vector &green, const SPM_settings &settings, double lambda) {
    auto & grid = settings.grid;
    auto V = grid.V;
    auto SVs = grid.SVs;
    auto U = grid.U;
    auto domegas = grid.domegas;

    int omega_dim = V.rows();
    int SV_dim = 0;
    double SV_tol = SVs(0)*settings.admm_params.sv_tol;
    for (int i = 0 ; i < SVs.size(); i++) {
        if (std::abs(SVs(i)) < SV_tol) {
            break;
        }
        SV_dim++;
    }

    double sum = -green(0) - green(green.size() - 1);

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
    double alpha = 0;
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
        double omegaHinvg = 0;
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
        double threshold = lambda/(settings.admm_params.mu_prime + (settings.admm_params.mu_prime == 0));
        for (int i = 0 ; i < SV_dim ; i++) {
            double val = x_prime(i) + u_prime(i);
            double positive = val - threshold;
            double negative = -val - threshold;
            z_prime(i) = positive*(positive > 0) - negative*(negative > 0);
        }
        temp = V*x_prime + u;
        for (int i = 0 ; i < omega_dim ; i++) {
            z(i) = temp(i)*(temp(i) >= 0);
        }
        u_prime += x_prime - z_prime;
        u += V*x_prime - z;
    }
    return V*x_prime;
}

spm::Vector spm::admm_minimize_raw(const Vector & y_prime, const Vector & SVs, const Vector & H_inv, const Vector & omega_prime, const Matrix & V, double alpha, double lambda, double mu, double mu_prime, double sum, bool fix_sum, int max_iter) {
    auto SV_dim = SVs.size();
    auto omega_dim = V.cols();
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
        double omegaHinvg = 0;
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
        double threshold = lambda/(mu_prime + (mu_prime == 0));
        for (int i = 0 ; i < SV_dim ; i++) {
            double val = x_prime(i) + u_prime(i);
            double positive = val - threshold;
            double negative = -val - threshold;
            z_prime(i) = positive*(positive > 0) - negative*(negative > 0);
        }
        temp = V*x_prime + u;
        for (int i = 0 ; i < omega_dim ; i++) {
            z(i) = temp(i)*(temp(i) >= 0);
        }
        u_prime += x_prime - z_prime;
        u += V*x_prime - z;
    }
    return V*x_prime;
}

spm::Vector spm::calculate_lambda_errs(const Vector &lambdas, const Vector & green, SPM_settings &settings) {
    Vector chi_sqr = Vector::Zero(lambdas.size());

    auto & grid = settings.grid;
    auto V = grid.V;
    auto SVs = grid.SVs;
    auto U = grid.U;
    auto domegas = grid.domegas;

    int omega_dim = V.rows();
    int SV_dim = 0;
    double SV_tol = SVs(0)*settings.admm_params.sv_tol;
    for (int i = 0 ; i < SVs.size(); i++) {
        if (std::abs(SVs(i)) < SV_tol) {
            break;
        }
        SV_dim++;
    }

    double sum = -green(0) - green(green.size() - 1);

    Vector y_prime = U.transpose()*green;
    Vector omega_prime = V.transpose()*domegas;

    SVs.conservativeResize(SV_dim);
    U.conservativeResize(Eigen::NoChange, SV_dim);
    V.conservativeResize(Eigen::NoChange, SV_dim);
    y_prime.conservativeResize(SV_dim);
    omega_prime.conservativeResize(SV_dim);

    Vector H_inv = Vector::Zero(SV_dim);
    double alpha = 0;
    for (int i = 0 ; i < SV_dim; i++) {
        H_inv(i) = 1/(settings.admm_params.mu + settings.admm_params.mu_prime + (SVs(i) * SVs(i)));
        alpha += omega_prime(i)*omega_prime(i)*H_inv(i);
    }
    alpha = 1/alpha;


    for (int i = 0 ; i < lambdas.size() ; i++) {
        logger::log->info("Calculating lambda errs for {}", lambdas(i));
        Vector rho = admm_minimize_raw(y_prime, SVs, H_inv, omega_prime, V, alpha, lambdas(i),
                                settings.admm_params.mu, settings.admm_params.mu_prime, sum, settings.admm_params.fix_sum, settings.admm_params.max_iters);
        auto & kernel = settings.grid.kernel;
        Vector green_rc = kernel*rho;
        //logger::log->info("kernel dims : {}, {}", kernel.rows(), kernel.cols());
        //logger::log->info("green_rc dims : {}, {}", green_rc.rows(), green_rc.cols());
        //logger::log->info("green dims : {}, {}", greens.rows(), greens.cols());
        chi_sqr(i) = (green - settings.grid.kernel*rho).squaredNorm()/2;
        logger::log->info("Chi_sqr : {}", chi_sqr(i));
    }
    return chi_sqr;
}
