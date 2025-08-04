#include <h5pp/h5pp.h>
#include "cli.h"
#include "spm.h"



int main(int argc, char * argv[]) {
    cli::parse(argc, argv);
    spm::run_spm(cli::settings_path);
    //helper::launch_annealing(cli::settings_path);
    return 0;
}