#include "wsvt/console_ops.hpp"

#include <iostream>
#include <stdexcept>

namespace wsvt {

void prColor(const std::string& word, const std::string& color_type) {
    std::string start_c;
    const std::string end_c = "\033[00m";
    if (color_type == "red") {
        start_c = "\033[91m";
    } else if (color_type == "green") {
        start_c = "\033[92m";
    } else if (color_type == "yellow") {
        start_c = "\033[93m";
    } else if (color_type == "light_purple") {
        start_c = "\033[94m";
    } else if (color_type == "purple") {
        start_c = "\033[95m";
    } else if (color_type == "cyan") {
        start_c = "\033[96m";
    } else if (color_type == "light_gray") {
        start_c = "\033[97m";
    } else if (color_type == "black") {
        start_c = "\033[98m";
    } else {
        throw std::invalid_argument("prColor: unknown color type '" + color_type + "'");
    }
    std::cout << start_c << word << end_c << std::endl;
}

}
