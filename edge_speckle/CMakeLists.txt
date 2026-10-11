cmake_minimum_required(VERSION 3.16)
project(speckle_edge LANGUAGES CXX CUDA)
set(EDGE_CUDA_ARCH "87" CACHE STRING "CUDA SM target; reviewed device is Orin SM87")
find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(edge_kernels SHARED native/edge_kernels.cu)
# CMake3.16 on Ubuntu20.04 cannot infer CUDA17 flags; CUDA11.4 accepts this explicitly.
target_compile_options(edge_kernels PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--std=c++17;--fmad=false;-arch=sm_${EDGE_CUDA_ARCH}>")
add_library(edge_spline SHARED native/edge_spline.cpp)
target_compile_features(edge_spline PRIVATE cxx_std_17)
target_compile_options(edge_spline PRIVATE -O3)
target_link_libraries(edge_spline PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(edge_kernels edge_spline PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/speckle_edge/_native")

# Historical reproductions are opt-in and use independent translation units.
foreach(version match optimized cached)
    add_library(wxst_${version} SHARED EXCLUDE_FROM_ALL native/legacy_${version}.cu)
    set_target_properties(wxst_${version} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/baselines/legacy_wxst")
    target_compile_options(wxst_${version} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3;--std=c++17;--fmad=false;-arch=sm_${EDGE_CUDA_ARCH}>")
endforeach()
