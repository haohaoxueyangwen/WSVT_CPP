include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(wsvt_tests
    test_common.cpp
    test_pyramid.cpp
    test_template_window.cpp
    test_umpa_physical_fit.cpp
    test_umpa_ddf_kernel.cpp
    test_wsvt_umpa_refine.cpp
    test_wsvt_pipeline.cpp
    test_fixed_spatial_cost.cpp
    test_guarded_subpixel.cpp
    test_directional_darkfield.cpp
    test_wavelet.cpp
    test_wavelet_hwd.cpp
    test_phase_recovery.cpp
    test_search_pruning.cpp
    test_wxst_smoke.cpp
)

target_link_libraries(wsvt_tests PRIVATE wsvt_core Catch2::Catch2WithMain)

add_executable(wsvt_umpa_fixture umpa_fixture_cli.cpp)
target_link_libraries(wsvt_umpa_fixture PRIVATE wsvt_core)

add_executable(wsvt_umpa_image_fixture umpa_image_fixture_cli.cpp)
target_link_libraries(wsvt_umpa_image_fixture PRIVATE wsvt_core)

add_executable(wsvt_umpa_official_parity umpa_official_parity_cli.cpp)
target_link_libraries(wsvt_umpa_official_parity PRIVATE wsvt_core)

add_executable(wsvt_umpa_official_minimizer umpa_official_minimizer_cli.cpp)
target_link_libraries(wsvt_umpa_official_minimizer PRIVATE wsvt_core)

add_executable(wsvt_umpa_ddf_kernel_parity_cli umpa_ddf_kernel_parity_cli.cpp)
target_link_libraries(wsvt_umpa_ddf_kernel_parity_cli PRIVATE wsvt_core)

add_executable(wsvt_umpa_ddf_kernel_controls_cli umpa_ddf_kernel_controls_cli.cpp)
target_link_libraries(wsvt_umpa_ddf_kernel_controls_cli PRIVATE wsvt_core)

add_executable(wsvt_umpa_ddf_kernel_optimizer_cli umpa_ddf_kernel_optimizer_cli.cpp)
target_link_libraries(wsvt_umpa_ddf_kernel_optimizer_cli PRIVATE wsvt_core)

add_executable(wsvt_objective_rank wsvt_objective_rank_cli.cpp)
target_link_libraries(wsvt_objective_rank PRIVATE wsvt_core)

add_executable(wsvt_fixed_support_fixture fixed_support_fixture_cli.cpp)
target_link_libraries(wsvt_fixed_support_fixture PRIVATE wsvt_core)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)
catch_discover_tests(wsvt_tests)
