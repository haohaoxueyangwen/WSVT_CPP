cmake_minimum_required(VERSION 3.21)
project(WSVT_CPP VERSION 0.1.0 LANGUAGES C CXX)

# ── C++ standard ──────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ───────────────────────────────────────────────────
option(WSVT_WITH_HDF5   "Enable HDF5 IO support"                          ON)
option(WSVT_WITH_OPENCV "Enable OpenCV-based slope estimate support"       ON)
option(WSVT_WITH_FFTW   "Enable FFTW for fast Frankot-Chellappa recovery" ON)
option(BUILD_SHARED_LIBS "Build wsvt_core as a shared library"            OFF)
option(WSVT_BUILD_TESTS "Build unit tests (requires Catch2)"              OFF)
option(WSVT_BUILD_BENCH "Build micro-benchmarks (requires nanobench)"     OFF)
option(WSVT_DISABLE_AUTOVEC "Disable compiler auto-vectorization for SIMD ablation builds" OFF)
option(WSVT_VEC_REPORT "Emit compiler vectorization diagnostics for wsvt_core" OFF)
option(WSVT_SEARCH_STATIC_SCHEDULE "Use static scheduling in the WSVT search ablation" OFF)
option(WSVT_NATIVE_ARCH "Enable -march=native in Release builds (non-portable)" OFF)

# ── Core library ──────────────────────────────────────────────
add_library(wsvt_core
    src/core.cpp
    src/euclidean_dist.cpp
    src/image_ops.cpp
    src/template_window.cpp
    src/phase_recovery.cpp
    src/pyramid.cpp
    src/wavelet_ops.cpp
    src/filter_ops.cpp
    src/io_h5.cpp
    src/io_json.cpp
    src/io_image.cpp
    src/align_ops.cpp
    src/crop_ops.cpp
    src/console_ops.cpp
    src/solver_utils.cpp
    src/umpa_physical_fit.cpp
    src/umpa_ddf_kernel.cpp
    src/umpa_ddf_kernel_optimizer.cpp
    src/umpa_ddf_author_fit.cpp
    src/directional_darkfield.cpp
    src/wsvt_umpa_refine.cpp
    src/wsvt_pipeline.cpp
    src/wxst_pipeline.cpp
)

target_compile_features(wsvt_core PUBLIC cxx_std_20)
add_library(wsvt_ddf_author SHARED src/umpa_ddf_author_c_api.cpp)
target_link_libraries(wsvt_ddf_author PRIVATE wsvt_core)
if(WSVT_SEARCH_STATIC_SCHEDULE)
    target_compile_definitions(wsvt_core PRIVATE WSVT_SEARCH_STATIC_SCHEDULE)
endif()

target_include_directories(wsvt_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Symbol visibility: hide by default, export via WSVT_API macro
set_target_properties(wsvt_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)

if(BUILD_SHARED_LIBS)
    target_compile_definitions(wsvt_core PUBLIC WSVT_SHARED_BUILD)
endif()

# ── Compile flags ─────────────────────────────────────────────
if(MSVC)
    target_compile_options(wsvt_core PRIVATE /utf-8 /W4)
    # Release flags handled by CMAKE_BUILD_TYPE
else()
    target_compile_options(wsvt_core PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow
        $<$<CONFIG:Release>:-O3 -funroll-loops>
        $<$<CONFIG:Release>:-fno-math-errno -ffinite-math-only -fno-signed-zeros -fno-trapping-math>
    )
    if(WSVT_NATIVE_ARCH)
        target_compile_options(wsvt_core PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
endif()

if(WSVT_DISABLE_AUTOVEC)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(wsvt_core PRIVATE -fno-tree-vectorize -fno-tree-slp-vectorize)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(wsvt_core PRIVATE -fno-vectorize -fno-slp-vectorize)
    else()
        message(WARNING "WSVT_DISABLE_AUTOVEC is not configured for compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

# DDF uses NaNs as undefined physical outputs and validates non-finite inputs.
# These overrides are source-local; existing WSVT/WXST kernel flags stay intact.
if(NOT MSVC)
    set_source_files_properties(src/umpa_ddf_author_fit.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-finite-math-only;-fsigned-zeros;-ffp-contract=off")
    set_source_files_properties(src/directional_darkfield.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-finite-math-only;-fsigned-zeros;-ffp-contract=off")
endif()

if(WSVT_VEC_REPORT)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(wsvt_core PRIVATE
            -fopt-info-vec-all=${CMAKE_BINARY_DIR}/vec-all.log
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(wsvt_core PRIVATE
            -Rpass=loop-vectorize
            -Rpass-missed=loop-vectorize
        )
    else()
        message(WARNING "WSVT_VEC_REPORT is not configured for compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endif()

# ── OpenMP ────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)
target_link_libraries(wsvt_core PUBLIC OpenMP::OpenMP_CXX)

# ── Conda / non-standard library paths ───────────────────────
set(WSVT_RUNTIME_LIB_DIRS "")
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND WSVT_RUNTIME_LIB_DIRS "$ENV{CONDA_PREFIX}/lib")
endif()

# ── HDF5 (optional) ──────────────────────────────────────────
if(WSVT_WITH_HDF5)
    find_package(HDF5 QUIET COMPONENTS C)
    if(HDF5_FOUND)
        target_compile_definitions(wsvt_core PRIVATE WSVT_HAS_HDF5)
        target_include_directories(wsvt_core PRIVATE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(wsvt_core PUBLIC ${HDF5_LIBRARIES})
        message(STATUS "WSVT: HDF5 enabled")
    else()
        message(WARNING "WSVT: HDF5 not found, io_h5 read/write will throw at runtime")
    endif()
endif()

# ── OpenCV (optional) ────────────────────────────────────────
if(WSVT_WITH_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs video)
    if(OpenCV_FOUND)
        target_compile_definitions(wsvt_core PRIVATE WSVT_HAS_OPENCV)
        target_include_directories(wsvt_core PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(wsvt_core PUBLIC ${OpenCV_LIBS})
        if(OpenCV_DIR)
            get_filename_component(WSVT_OPENCV_CMAKE_DIR "${OpenCV_DIR}" DIRECTORY)
            get_filename_component(WSVT_OPENCV_LIB_DIR "${WSVT_OPENCV_CMAKE_DIR}" DIRECTORY)
            list(PREPEND WSVT_RUNTIME_LIB_DIRS "${WSVT_OPENCV_LIB_DIR}")
            find_library(WSVT_TIFF_LIBRARY NAMES tiff PATHS "${WSVT_OPENCV_LIB_DIR}" NO_DEFAULT_PATH)
            find_library(WSVT_DEFLATE_LIBRARY NAMES deflate PATHS "${WSVT_OPENCV_LIB_DIR}" NO_DEFAULT_PATH)
            if(WSVT_TIFF_LIBRARY)
                target_link_libraries(wsvt_core PUBLIC "${WSVT_TIFF_LIBRARY}")
            endif()
            if(WSVT_DEFLATE_LIBRARY)
                target_link_libraries(wsvt_core PUBLIC "${WSVT_DEFLATE_LIBRARY}")
            endif()
        endif()
        message(STATUS "WSVT: OpenCV enabled")
    else()
        message(WARNING "WSVT: OpenCV not found, use_estimate=true will throw at runtime")
    endif()
endif()

if(WSVT_RUNTIME_LIB_DIRS)
    list(REMOVE_DUPLICATES WSVT_RUNTIME_LIB_DIRS)
    target_link_directories(wsvt_core BEFORE PUBLIC ${WSVT_RUNTIME_LIB_DIRS})
endif()

# ── FFTW3 (optional) ─────────────────────────────────────────
if(WSVT_WITH_FFTW)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3F QUIET fftw3f)
    endif()
    if(NOT FFTW3F_FOUND)
        find_library(FFTW3F_LIB NAMES fftw3f)
        find_path(FFTW3F_INC NAMES fftw3.h)
        if(FFTW3F_LIB AND FFTW3F_INC)
            set(FFTW3F_FOUND TRUE)
            set(FFTW3F_LIBRARIES ${FFTW3F_LIB})
            set(FFTW3F_INCLUDE_DIRS ${FFTW3F_INC})
        endif()
    endif()
    if(FFTW3F_FOUND)
        target_compile_definitions(wsvt_core PRIVATE WSVT_HAS_FFTW)
        target_include_directories(wsvt_core PRIVATE ${FFTW3F_INCLUDE_DIRS})
        target_link_libraries(wsvt_core PUBLIC ${FFTW3F_LIBRARIES})
        find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads fftw3f_omp)
        if(FFTW3F_THREADS_LIB)
            target_link_libraries(wsvt_core PUBLIC ${FFTW3F_THREADS_LIB})
            message(STATUS "WSVT: FFTW3 threading support enabled")
        endif()
        if(FFTW3F_LIBRARY_DIRS)
            target_link_directories(wsvt_core PUBLIC ${FFTW3F_LIBRARY_DIRS})
        endif()
        message(STATUS "WSVT: FFTW3 (single precision) enabled")
    else()
        message(WARNING "WSVT: FFTW3 not found, using O(n^4) brute-force DFT fallback")
    endif()
endif()

# ── CLI executable ────────────────────────────────────────────
add_executable(wsvt_cli src/main.cpp)
target_link_libraries(wsvt_cli PRIVATE wsvt_core)
add_executable(wsvt_ddf_cli src/directional_darkfield_main.cpp)
target_link_libraries(wsvt_ddf_cli PRIVATE wsvt_core)

# ── Install / Export ──────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS wsvt_core wsvt_cli wsvt_ddf_cli wsvt_ddf_author
    EXPORT wsvtTargets
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/wsvt
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT wsvtTargets
    FILE wsvtTargets.cmake
    NAMESPACE wsvt::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wsvt
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/wsvtConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/wsvtConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wsvt
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/wsvtConfigVersion.cmake
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/wsvtConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/wsvtConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/wsvt
)

# ── Tests ─────────────────────────────────────────────────────
if(WSVT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Benchmarks ────────────────────────────────────────────────
if(WSVT_BUILD_BENCH)
    add_subdirectory(bench)
endif()
