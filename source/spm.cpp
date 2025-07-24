#include "spm.h"


void spm::save_grid(Grid &grid, std::string filename) {
    h5pp::File grid_file(filename, h5pp::FileAccess::COLLISION_FAIL);
    grid_file.writeDataset(grid.SVs, "SVs");
    grid_file.writeDataset(grid.U, "U");
    grid_file.writeDataset(grid.V, "V");
    grid_file.writeDataset(grid.taus, "taus");
    grid_file.writeDataset(grid.omegas, "omegas");
    grid_file.writeDataset(grid.domegas, "domegas");
    grid_file.writeDataset(grid.n_taus, "n_taus");
    grid_file.writeDataset(grid.n_omegas, "n_omegas");
    grid_file.writeDataset(grid.beta, "beta");
}

spm::Grid spm::load_grid(const std::string filename) {
    h5pp::File grid_file(filename, h5pp::FileAccess::READONLY);
    Grid grid;
    grid.SVs = grid_file.readDataset<Vector>("SVs");
    grid.U   = grid_file.readDataset<Matrix>("U");
    grid.V   = grid_file.readDataset<Matrix>("V");

    grid.taus    = grid_file.readDataset<Vector>("SVs");
    grid.omegas  = grid_file.readDataset<Vector>("SVs");
    grid.domegas = grid_file.readDataset<Vector>("SVs");

    grid.n_taus   = grid_file.readDataset<int>("SVs");
    grid.n_omegas = grid_file.readDataset<int>("SVs");
    grid.beta     = grid_file.readDataset<double>("SVs");

    return grid;
}


spm::Grid spm::generate_grid(Vector omegas, Vector domegas, int n_taus, double beta) {
    assert(n_taus > 0);
    assert(beta > 0);
    logger::log->info("Creating grid...");
    Vector taus = Vector::LinSpaced(n_taus, 0, beta);
    int n_omegas = omegas.size();
    Matrix kernel = Matrix::Zero(n_taus, n_omegas);
    for (int it = 0; it < n_taus; it++) {
        for (int iw = 0 ; iw < n_omegas; iw++) {
            double val = -domegas[iw]*std::exp(-taus[it]*omegas[iw])/(1 + std::exp(-beta*omegas[iw]));
            //logger::log->info("Tau : {}, Omega : {}, K(tau, omega) : {}", taus[it], omegas[iw], val);
            kernel(it, iw) = val;
        }
    }
    logger::log->info("Performing SVD decomposition...");
    Eigen::BDCSVD svd(kernel, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Grid grid { .SVs = svd.singularValues(), .U = svd.matrixU(), .V = svd.matrixV(),
                .taus = taus, .omegas = omegas, .domegas = domegas,
                .n_taus = n_taus, .n_omegas = n_omegas, .beta = beta};
    logger::log->info("Done");
    return grid;
}


void spm::test_spm() {
    logger::log = spdlog::stdout_color_mt("Spectral", spdlog::color_mode::always);
    logger::log->set_level(static_cast<spdlog::level::level_enum>(0));
    double omega_min = -1.0;
    double omega_max = 1.0;
    int n_omegas = 2;
    Vector omegas = Vector::LinSpaced(n_omegas, omega_min, omega_max);
    Vector domegas = Vector::Ones(n_omegas);
    domegas *= (omega_max - omega_min)/(n_omegas - 1);
    int n_taus = 2;
    double beta = 1.0;
    auto grid = generate_grid(omegas, domegas, n_taus, beta);
}
