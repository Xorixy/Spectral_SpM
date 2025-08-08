#include "spm.h"




spm::Grid spm::generate_grid(Vector omegas, Vector domegas, int n_taus, double beta, double recursion_tolerance) {
    assert(n_taus > 0);
    assert(beta > 0);
    logger::log->info("Creating grid...");
    bool centrosymmetric = true;
    for (int i = 0; i < omegas.size(); i++) {
        if (omegas(i) != -omegas(omegas.size() - i - 1)) {
            centrosymmetric = false;
            break;
        }
    }
    logger::log->info("Centrosymmetric : {}", centrosymmetric);
    Vector taus = symmetric_linspace(n_taus, static_cast<Scalar>(beta)/2, static_cast<Scalar>(beta)/2);
    int n_omegas = omegas.size();
    Matrix kernel = Matrix::Zero(n_taus, n_omegas);
    for (int it = 0; it < n_taus; it++) {
        for (int iw = 0 ; iw < n_omegas; iw++) {
            Scalar val = -domegas[iw]*std::exp(static_cast<Scalar>(beta)*omegas[iw] - taus[it]*omegas[iw])/(1 + std::exp(static_cast<Scalar>(beta)*omegas[iw]));
            //logger::log->info("Tau : {}, Omega : {}, K(tau, omega) : {}", taus[it], omegas[iw], val);
            kernel(it, iw) = val;
        }
    }
    SVD svd;
    if (centrosymmetric) {



    } else {
        logger::log->info("Performing direct SVD decomposition (recursion tolerance : {})...", recursion_tolerance);
        svd = recursive_svd(kernel.cast<LScalar>(), recursion_tolerance);
    }
    Grid grid { .SVs = svd.SVs.cast<Scalar>(), .U = svd.U.cast<Scalar>(), .V = svd.V.cast<Scalar>(),
                .taus = taus, .omegas = omegas, .domegas = domegas,
                .n_taus = n_taus, .n_omegas = n_omegas, .beta = beta,
                .kernel = kernel };
    logger::log->info("Done");
    //std::cout << grid.SVs << "\n\n";

    return grid;
}

spm::Vector spm::green_from_spectral(const Vector &spectral, Grid &grid) {
    return grid.U*grid.SVs.asDiagonal()*grid.V.transpose()*spectral;
}




void spm::run_spm(std::string settings_path) {
    test_centrosymmetric();
    return;
    auto [settings, green] = io::load_settings(settings_path);
    if (settings.debug.test_spectral) {
        logger::log->info("Spectral debug option enabled. Loading spectral function from file...");
        green = green_from_spectral(io::load_spectral(settings.debug.spectral_file), settings.grid);
        io::save_green(settings.output_path, green);
    }
    double l_min = std::log10(settings.admm_params.lambda_min);
    double l_max = std::log10(settings.admm_params.lambda_max);
    int N = std::round(l_max - l_min) + 1;
    N = std::round((settings.admm_params.lambda_res - 1)*(N - 1)) + N;
    Vector lambdas = Vector::LinSpaced(N, l_min, l_max);
    for (int i = 0 ; i < N; i++) {
        lambdas(i) = pow(10, lambdas(i));
    }
    double lambda_opt = 0;
    if (settings.debug.test_convergence) {
        Vector errors = admm_check_convergence(green, settings, settings.admm_params.lambda);
        io::save_vector(settings.output_path, errors, "errors");
    } else {
        if (settings.admm_params.override_lambda_opt) {
            lambda_opt = settings.admm_params.lambda;
            logger::log->info("Running single sim for lambda : {}", lambda_opt);
        } else {
            logger::log->info("Running {} lambda sims", N);
            Vector errors = calculate_lambda_errs(lambdas, green, settings);
            double a = (errors(errors.size() - 1) - errors(0))/(std::log(lambdas(lambdas.size() - 1)) - std::log(lambdas(0)));
            double div_max = std::numeric_limits<double>::min();
            for (int i = 0 ; i < errors.size(); i++) {
                double div = a*(std::log(lambdas(i)) - std::log(lambdas(0))) + errors(0);
                if (div / errors(i) > div_max) {
                    div_max = div / errors(i);
                    lambda_opt = lambdas(i);
                }
            }
            logger::log->info("Lambda opt: {}", lambda_opt);
            io::save_vector(settings.output_path, lambdas, "lambdas");
            io::save_vector(settings.output_path, errors, "errors");
        }


        Vector spectral = admm_minimize(green, settings, lambda_opt);
        Vector green_rc = settings.grid.kernel*spectral;
        io::save_spectral(settings.output_path, spectral);
        io::save_vector(settings.output_path, green_rc, "green_rc");
    }
}
