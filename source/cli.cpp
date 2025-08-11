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

#include "cli.h"


int cli::parse(const int argc, char *argv[]) {
    CLI::App app;
    app.description("Sparse matrix method for spectral functions");
    app.get_formatter()->column_width(90);
    app.option_defaults()->always_capture_default();
    app.allow_extras(false);

    /* clang-format off */
    app.add_option("--settings",settings_path,"Path to a .json file with settings");

    /* clang-format on */
    CLI11_PARSE(app, argc, argv);

    // Init
    logger::log = spdlog::stdout_color_mt("Spectral", spdlog::color_mode::always);
    logger::log->set_level(static_cast<spdlog::level::level_enum>(0));

    return 0;
}