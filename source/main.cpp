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

#include <h5pp/h5pp.h>
#include "cli.h"
#include "spm.h"



int main(int argc, char * argv[]) {
    cli::parse(argc, argv);
    spm::run_spm(cli::settings_path);
    //helper::launch_annealing(cli::settings_path);
    return 0;
}