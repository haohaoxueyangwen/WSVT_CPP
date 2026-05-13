#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace wsvt {

struct Image2D {
    std::vector<float> data;
    std::size_t h;
    std::size_t w;
};

std::vector<std::string> list_image_files(const std::string& folder);
Image2D read_image_gray(const std::string& filename);

void save_img(
    const std::vector<float>& img,
    std::size_t h,
    std::size_t w,
    const std::string& filename);

}
