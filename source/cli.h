#pragma once
#include "spectral.h"
#include <CLI/CLI.hpp>
#include <fmt/core.h>
#include "log.h"

namespace cli {
    inline std::string settings_path;
    int parse(int argc, char *argv[]);
}