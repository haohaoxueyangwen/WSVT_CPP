#include "wsvt/io_image.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(WSVT_HAS_OPENCV)
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>
#endif

namespace wsvt {

namespace {

void write_u16(std::ofstream& os, std::uint16_t v) {
    const char b[2] = {
        static_cast<char>(v & 0xFF),
        static_cast<char>((v >> 8) & 0xFF)
    };
    os.write(b, 2);
}

void write_u32(std::ofstream& os, std::uint32_t v) {
    const char b[4] = {
        static_cast<char>(v & 0xFF),
        static_cast<char>((v >> 8) & 0xFF),
        static_cast<char>((v >> 16) & 0xFF),
        static_cast<char>((v >> 24) & 0xFF)
    };
    os.write(b, 4);
}

struct IfdEntry {
    std::uint16_t tag;
    std::uint16_t type;
    std::uint32_t count;
    std::uint32_t value_or_offset;
};

}

std::vector<std::string> list_image_files(const std::string& folder) {
    namespace fs = std::filesystem;
    const fs::path dir(folder);
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        throw std::runtime_error("image folder not found: " + folder);
    }
    std::vector<std::string> out;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (ext == ".tif" || ext == ".tiff" || ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".webp") {
            out.push_back(e.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    if (out.empty()) {
        throw std::runtime_error("no image files in folder: " + folder);
    }
    return out;
}

Image2D read_image_gray(const std::string& filename) {
#if defined(WSVT_HAS_OPENCV)
    // Suppress OpenCV TIFF warnings via OpenCV logging API (no extra link deps).
    try {
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    } catch (...) {
        // If logging API unavailable, proceed without throwing — warnings may appear.
    }
    const cv::Mat img = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("failed to read image: " + filename);
    }
    cv::Mat gray;
    if (img.channels() == 1) {
        gray = img;
    } else {
        std::vector<cv::Mat> chs;
        cv::split(img, chs);
        gray = chs[0];
    }
    cv::Mat gray32;
    gray.convertTo(gray32, CV_32F);
    Image2D out;
    out.h = static_cast<std::size_t>(gray32.rows);
    out.w = static_cast<std::size_t>(gray32.cols);
    out.data.resize(out.h * out.w, 0.0f);
    for (std::size_t y = 0; y < out.h; ++y) {
        const float* row = gray32.ptr<float>(static_cast<int>(y));
        for (std::size_t x = 0; x < out.w; ++x) {
            out.data[y * out.w + x] = row[x];
        }
    }
    return out;
#else
    throw std::runtime_error("read_image_gray requires OpenCV (WSVT_HAS_OPENCV)");
#endif
}

void save_img(
    const std::vector<float>& img,
    std::size_t h,
    std::size_t w,
    const std::string& filename) {
    if (img.size() != h * w) {
        throw std::invalid_argument("save_img size mismatch");
    }

    std::ofstream os(filename, std::ios::binary | std::ios::trunc);
    if (!os) {
        throw std::runtime_error("failed to open image file for write");
    }

    const std::uint32_t ifd_offset = 8;
    const std::uint16_t entry_count = 13;
    const std::uint32_t ifd_size = 2 + static_cast<std::uint32_t>(entry_count) * 12 + 4;
    const std::uint32_t xres_offset = ifd_offset + ifd_size;
    const std::uint32_t yres_offset = xres_offset + 8;
    const std::uint32_t image_offset = yres_offset + 8;
    const std::uint32_t byte_count = static_cast<std::uint32_t>(img.size() * sizeof(float));

    os.put('I');
    os.put('I');
    write_u16(os, 42);
    write_u32(os, ifd_offset);

    os.seekp(ifd_offset, std::ios::beg);
    write_u16(os, entry_count);

    // IFD entries must be sorted by tag number
    const std::vector<IfdEntry> entries = {
        {256, 4, 1, static_cast<std::uint32_t>(w)},       // ImageWidth
        {257, 4, 1, static_cast<std::uint32_t>(h)},       // ImageLength
        {258, 3, 1, 32},                                   // BitsPerSample
        {259, 3, 1, 1},                                    // Compression (None)
        {262, 3, 1, 1},                                    // PhotometricInterpretation (MinIsBlack)
        {273, 4, 1, image_offset},                         // StripOffsets
        {277, 3, 1, 1},                                    // SamplesPerPixel
        {278, 4, 1, static_cast<std::uint32_t>(h)},       // RowsPerStrip
        {279, 4, 1, byte_count},                           // StripByteCounts
        {282, 5, 1, xres_offset},                          // XResolution (RATIONAL)
        {283, 5, 1, yres_offset},                          // YResolution (RATIONAL)
        {296, 3, 1, 1},                                    // ResolutionUnit (No unit)
        {339, 3, 1, 3}                                     // SampleFormat (IEEE float)
    };

    for (const auto& e : entries) {
        write_u16(os, e.tag);
        write_u16(os, e.type);
        write_u32(os, e.count);
        write_u32(os, e.value_or_offset);
    }
    write_u32(os, 0);

    os.seekp(xres_offset, std::ios::beg);
    write_u32(os, 1);
    write_u32(os, 1);
    os.seekp(yres_offset, std::ios::beg);
    write_u32(os, 1);
    write_u32(os, 1);

    os.seekp(image_offset, std::ios::beg);
    os.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(byte_count));
    os.close();
}

}
