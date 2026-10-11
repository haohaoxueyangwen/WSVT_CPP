#include "wsvt/io_h5.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>

#if defined(WSVT_HAS_HDF5)
#include <hdf5.h>
#endif

namespace wsvt {

namespace {

void check_shape_and_data(const NdArrayF32& arr) {
    if (arr.shape.empty()) {
        throw std::invalid_argument("hdf5 data shape must not be empty");
    }
    std::size_t n = 1;
    for (const auto s : arr.shape) {
        if (s == 0) {
            throw std::invalid_argument("hdf5 shape contains zero");
        }
        n *= s;
    }
    if (n != arr.data.size()) {
        throw std::invalid_argument("hdf5 data size mismatch");
    }
}

}

void write_h5(
    const std::string& result_path,
    const std::string& file_name,
    const std::vector<H5ItemF32>& data_dict,
    int compress_level) {
#if defined(WSVT_HAS_HDF5)
    if (compress_level < 0 || compress_level > 9) {
        throw std::invalid_argument("hdf5 deflate level must be in [0, 9]");
    }

    std::filesystem::path out_dir(result_path);
    if (!std::filesystem::exists(out_dir)) {
        std::filesystem::create_directories(out_dir);
    }
    const std::filesystem::path file_path = out_dir / (file_name + ".hdf5");

    const hid_t file_id = H5Fcreate(file_path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("failed to create hdf5 file");
    }

    for (const auto& item : data_dict) {
        check_shape_and_data(item.value);

        std::vector<hsize_t> dims(item.value.shape.size(), 0);
        for (std::size_t i = 0; i < item.value.shape.size(); ++i) {
            dims[i] = static_cast<hsize_t>(item.value.shape[i]);
        }
        const hid_t space_id = H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr);
        if (space_id < 0) {
            H5Fclose(file_id);
            throw std::runtime_error("failed to create dataspace");
        }

        hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
        if (plist < 0) {
            H5Sclose(space_id);
            H5Fclose(file_id);
            throw std::runtime_error("failed to create property list");
        }
        if (compress_level > 0) {
            if (H5Pset_chunk(plist, static_cast<int>(dims.size()), dims.data()) < 0) {
                H5Pclose(plist);
                H5Sclose(space_id);
                H5Fclose(file_id);
                throw std::runtime_error("failed to set hdf5 chunk layout");
            }
            if (H5Pset_deflate(plist, static_cast<unsigned>(compress_level)) < 0) {
                H5Pclose(plist);
                H5Sclose(space_id);
                H5Fclose(file_id);
                throw std::runtime_error("failed to set hdf5 deflate filter");
            }
        }

        const hid_t dset_id = H5Dcreate2(
            file_id, item.key.c_str(), H5T_IEEE_F32LE, space_id,
            H5P_DEFAULT, plist, H5P_DEFAULT);
        if (dset_id < 0) {
            H5Pclose(plist);
            H5Sclose(space_id);
            H5Fclose(file_id);
            throw std::runtime_error("failed to create dataset");
        }

        const herr_t wr = H5Dwrite(
            dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, item.value.data.data());
        H5Dclose(dset_id);
        H5Pclose(plist);
        H5Sclose(space_id);
        if (wr < 0) {
            H5Fclose(file_id);
            throw std::runtime_error("failed to write dataset");
        }
    }
    H5Fclose(file_id);
    std::cout << "result hdf5 file : " << file_name << ".hdf5 saved" << std::endl;
#else
    (void)result_path;
    (void)file_name;
    (void)data_dict;
    throw std::runtime_error("hdf5.h not found, write_h5 unavailable");
#endif
}

NdArrayF32 read_h5(
    const std::string& file_path,
    const std::string& key_name,
    bool print_key) {
#if defined(WSVT_HAS_HDF5)
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("Wrong file path");
    }

    const hid_t file_id = H5Fopen(file_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("failed to open hdf5 file");
    }

    if (print_key) {
        H5G_info_t ginfo {};
        if (H5Gget_info(file_id, &ginfo) >= 0) {
            std::cout << "Keys: [";
            for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
                ssize_t name_len = H5Lget_name_by_idx(file_id, ".", H5_INDEX_NAME, H5_ITER_INC, i, nullptr, 0, H5P_DEFAULT);
                if (name_len > 0) {
                    std::string name(static_cast<std::size_t>(name_len), '\0');
                    H5Lget_name_by_idx(file_id, ".", H5_INDEX_NAME, H5_ITER_INC, i, name.data(), static_cast<std::size_t>(name_len + 1), H5P_DEFAULT);
                    std::cout << name;
                    if (i + 1 < ginfo.nlinks) {
                        std::cout << ", ";
                    }
                }
            }
            std::cout << "]" << std::endl;
        }
    }

    const hid_t dset_id = H5Dopen2(file_id, key_name.c_str(), H5P_DEFAULT);
    if (dset_id < 0) {
        H5Fclose(file_id);
        throw std::runtime_error("dataset key not found");
    }

    const hid_t space_id = H5Dget_space(dset_id);
    const int ndims = H5Sget_simple_extent_ndims(space_id);
    if (ndims <= 0) {
        H5Sclose(space_id);
        H5Dclose(dset_id);
        H5Fclose(file_id);
        throw std::runtime_error("invalid dataset dims");
    }

    std::vector<hsize_t> dims(static_cast<std::size_t>(ndims), 0);
    H5Sget_simple_extent_dims(space_id, dims.data(), nullptr);
    std::vector<std::size_t> shape(static_cast<std::size_t>(ndims), 0);
    std::size_t n = 1;
    for (int i = 0; i < ndims; ++i) {
        shape[static_cast<std::size_t>(i)] = static_cast<std::size_t>(dims[static_cast<std::size_t>(i)]);
        n *= shape[static_cast<std::size_t>(i)];
    }

    std::vector<float> data(n, 0.0f);
    const herr_t rd = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5Sclose(space_id);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    if (rd < 0) {
        throw std::runtime_error("failed to read dataset");
    }
    return NdArrayF32{std::move(shape), std::move(data)};
#else
    (void)file_path;
    (void)key_name;
    (void)print_key;
    throw std::runtime_error("hdf5.h not found, read_h5 unavailable");
#endif
}

}
