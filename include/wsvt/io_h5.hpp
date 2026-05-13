#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace wsvt {

struct NdArrayF32 {
    std::vector<std::size_t> shape;
    std::vector<float> data;
};

struct H5ItemF32 {
    std::string key;
    NdArrayF32 value;
};

void write_h5(
    const std::string& result_path,
    const std::string& file_name,
    const std::vector<H5ItemF32>& data_dict);

NdArrayF32 read_h5(
    const std::string& file_path,
    const std::string& key_name,
    bool print_key = false);

}
