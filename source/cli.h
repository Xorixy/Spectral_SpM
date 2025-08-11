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
#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include "log.h"

namespace cli {
    inline std::string settings_path { "../../settings/settings.json"};
    int parse(int argc, char *argv[]);
}