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