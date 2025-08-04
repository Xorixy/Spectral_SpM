#pragma once
#include <h5pp/h5pp.h>
#include <Eigen/Core>
#include "structs.h"
#include "spm.h"
#include "log.h"

namespace io {

    void save_grid(spm::Grid & grid, h5pp::File grid_file, bool save_svd = true);
    spm::Grid load_grid(std::string filename);

    std::pair<spm::Vector, double> load_greens_function(std::string filename);
    std::pair<spm::SPM_settings, spm::Vector> load_settings(std::string filename);

    void save_spectral(std::string filename, spm::Vector & spectral);
    void save_green(h5pp::File grid_file, spm::Vector & green);
    void save_green(std::string filename, spm::Vector &green);
    void save_vector(std::string filename, spm::Vector & vector, std::string name);
    std::pair<spm::Vector, spm::Vector> load_omegas(std::string filename);
    spm::Vector load_spectral(std::string filename);

    template<typename T>
        std::optional<T> parse_setting(const nlohmann::json &j, const std::string &name) {
        if (j.contains(name)) {
            try {
                return j[name].get<T>();
            } catch (std::exception &e) {
                return std::nullopt;
            }
        }
        return std::nullopt;
    }
}