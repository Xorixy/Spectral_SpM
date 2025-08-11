#include "io.h"

void io::save_grid(spm::Grid &grid, h5pp::File grid_file, bool save_svd) {
    if (save_svd) {
        grid_file.writeDataset(grid.SVs, "SVs");
        grid_file.writeDataset(grid.U, "U");
        grid_file.writeDataset(grid.V, "V");
        grid_file.writeDataset(grid.kernel, "kernel");
    }
    grid_file.writeDataset(grid.taus, "taus");
    grid_file.writeDataset(grid.omegas, "omegas");
    grid_file.writeDataset(grid.domegas, "domegas");
    grid_file.writeDataset(grid.n_taus, "n_taus");
    grid_file.writeDataset(grid.n_omegas, "n_omegas");
    grid_file.writeDataset(grid.beta, "beta");
}

spm::Grid io::load_grid(const std::string filename) {
    h5pp::File grid_file(filename, h5pp::FileAccess::READONLY);
    spm::Grid grid;
    grid.SVs = grid_file.readDataset<spm::Vector>("SVs");
    grid.U   = grid_file.readDataset<spm::Matrix>("U");
    grid.V   = grid_file.readDataset<spm::Matrix>("V");
    grid.kernel = grid_file.readDataset<spm::Matrix>("kernel");

    grid.taus    = grid_file.readDataset<spm::Vector>("taus");
    grid.omegas  = grid_file.readDataset<spm::Vector>("omegas");
    grid.domegas = grid_file.readDataset<spm::Vector>("domegas");

    grid.n_taus   = grid_file.readDataset<int>("n_taus");
    grid.n_omegas = grid_file.readDataset<int>("n_omegas");
    grid.beta     = grid_file.readDataset<double>("beta");

    return grid;
}

void io::save_spectral(std::string filename, spm::Vector &spectral) {
    h5pp::File outfile(filename, h5pp::FileAccess::READWRITE);
    outfile.writeDataset(spectral, "spectral");
}

void io::save_green(std::string filename, spm::Vector &green) {
    h5pp::File outfile(filename, h5pp::FileAccess::READWRITE);
    outfile.writeDataset(green, "green");
}

void io::save_vector(std::string filename, spm::Vector &vector, std::string name) {
    h5pp::File outfile(filename, h5pp::FileAccess::READWRITE);
    outfile.writeDataset(vector, name);
}

spm::Vector io::load_spectral(std::string filename) {
    h5pp::File infile(filename, h5pp::FileAccess::READONLY);
    return infile.readDataset<spm::Vector>("spectral");
}

std::pair<spm::Vector, spm::Vector> io::load_omegas(std::string filename) {
    h5pp::File infile(filename, h5pp::FileAccess::READONLY);
    spm::Vector omegas = infile.readDataset<spm::Vector>("omegas");
    spm::Vector domegas = infile.readDataset<spm::Vector>("domegas");
    return {omegas, domegas};
}

std::pair<spm::Vector, double> io::load_greens_function(std::string filename) {
    h5pp::File green_file(filename, h5pp::FileAccess::READONLY);
    std::vector<double> green_vec = green_file.readDataset<std::vector<double>>("green");
    double beta = green_file.readDataset<double>("beta");
    spm::Vector green = spm::Vector::Zero(green_vec.size());
    for (int i = 0 ; i < green_vec.size(); i++) {
        green(i) = green_vec[i];
    }
    return {green, beta};
}


std::pair<spm::SPM_settings, spm::Vector> io::load_settings(std::string filename) {
    logger::log->info("Loading settings from {}", filename);
    std::ifstream f(filename);
    spm::ADMM_params admm;
    spm::DebugSettings debug;
    auto j_settings = nlohmann::json::parse(f);
    logger::log->info("Loading Greens function...");
    std::string green_file = parse_setting<std::string>(j_settings, "green_file").value();
    spm::Vector green;
    double beta;
    auto load_green = load_greens_function(green_file);
    green = load_green.first;
    beta = load_green.second;
    int n_taus = green.size();
    logger::log->info("Greens function loaded. Number of imaginary time points : {}", n_taus);
    assert(n_taus > 1);
    spm::Grid grid;
    std::string output_path = parse_setting<std::string>(j_settings, "output_file").value();
    if (std::optional<std::string> input_grid_path = parse_setting<std::string>(j_settings, "input_grid_file");
        input_grid_path.has_value() && (input_grid_path.value() != "")) {
        //If this is true then we load an old grid
        //Otherwise we generate a new one
        logger::log->info("Path to old grid supplied, trying to load it...");
        grid = load_grid(input_grid_path.value());
        assert(grid.beta == beta);
        assert(grid.n_taus == n_taus);
    } else {
        logger::log->info("No old grid supplied. Generating new one from settings...");

        if (std::optional<std::string> omega_path = parse_setting<std::string>(j_settings, "input_omega_file");
            omega_path.has_value() && (omega_path != "")) {
            logger::log->info("Loading omegas from file {}", omega_path.value());
            auto [omegas, domegas] = load_omegas(omega_path.value());
            grid = spm::generate_grid(omegas, domegas, n_taus, beta);
        } else {
            double omega_min = parse_setting<double>(j_settings, "omega_min").value();
            double omega_max = parse_setting<double>(j_settings, "omega_max").value();
            int n_omegas = parse_setting<int>(j_settings, "n_omegas").value();
            double recursion_tolerance = parse_setting<double>(j_settings, "recursion_tolerance").value();
            logger::log->info("omega_min : {}, omega_max : {}, n_omegas : {}", omega_min, omega_max, n_omegas);
            assert(n_omegas > 1);
            spm::Vector omegas;
            if (omega_min == -omega_max) {
                logger::log->info("Creating symmetric omegas...");
                //omegas = spm::symmetric_linspace(n_omegas, omega_max);
                omegas = spm::Vector::LinSpaced(n_omegas, omega_min, omega_max);
            } else {
                logger::log->info("Creating non-symmetric omegas...");
                omegas = spm::Vector::LinSpaced(n_omegas, omega_min, omega_max);
            }
            spm::Scalar domega = (omega_max - omega_min) / (n_omegas - 1);
            spm::Vector domegas = spm::Vector::Ones(n_omegas)*domega;
            grid = spm::generate_grid(omegas, domegas, n_taus, beta, recursion_tolerance);
        }
    }
    bool save_svd = false;
    if (std::optional<bool> save_svd_set = parse_setting<bool>(j_settings, "save_svd");
        save_svd_set.has_value()) {
        save_svd = save_svd_set.value();
    }
    logger::log->info("Save svd : {}", save_svd);

    h5pp::File outfile(output_path, h5pp::FileAccess::REPLACE);
    save_grid(grid, outfile, save_svd);
    outfile.writeDataset(green, "green");
    {
        logger::log->info("Loading admm settings...");
        auto j_admm = j_settings["admm"];
        admm.lambda = parse_setting<double>(j_admm, "lambda").value();
        admm.sv_tol = parse_setting<double>(j_admm, "sv_tol").value();
        admm.max_iters = parse_setting<int>(j_admm, "max_iter").value();
        admm.mu = parse_setting<double>(j_admm, "mu").value();
        admm.mu_prime = parse_setting<double>(j_admm, "mu_prime").value();
        admm.fix_sum = parse_setting<bool>(j_admm, "fix_sum").value();
        admm.direct_inversion = parse_setting<bool>(j_admm, "direct_inversion").value();
        admm.lambda_min = parse_setting<double>(j_admm, "lambda_min").value();
        admm.lambda_max = parse_setting<double>(j_admm, "lambda_max").value();
        admm.lambda_res = parse_setting<double>(j_admm, "lambda_res").value();
        admm.override_lambda_opt = parse_setting<bool>(j_admm, "override_lambda_opt").value();
    }
    if (j_settings.contains("debug")) {
        auto j_debug = j_settings["debug"];
        debug.test_spectral = parse_setting<bool>(j_debug, "test_spectral").value();
        debug.spectral_file = parse_setting<std::string>(j_debug, "spectral_file").value();
        debug.test_convergence = parse_setting<bool>(j_debug, "test_convergence").value();
    }

    logger::log->info("Settings loaded");
    return {spm::SPM_settings {.grid = grid, .admm_params = admm, .debug = debug, .output_path = output_path}, green};
}