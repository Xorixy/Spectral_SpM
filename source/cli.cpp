#include "cli.h"


int cli::parse(const int argc, char *argv[]) {
    CLI::App app;
    app.description("Quantum quadrupling worm");
    app.get_formatter()->column_width(90);
    app.option_defaults()->always_capture_default();
    app.allow_extras(false);

    /* clang-format off */
    app.add_option("--settings",settings_path,"Path to a .json file with sim settings");

    /* clang-format on */
    CLI11_PARSE(app, argc, argv);

    // Init
    logger::log = spdlog::stdout_color_mt("Spectral", spdlog::color_mode::always);
    logger::log->set_level(static_cast<spdlog::level::level_enum>(0));

    return 0;
}