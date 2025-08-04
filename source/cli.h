#pragma once
#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include "log.h"

namespace cli {
    inline std::string settings_path { "../../settings/settings.json"};
    int parse(int argc, char *argv[]);
}