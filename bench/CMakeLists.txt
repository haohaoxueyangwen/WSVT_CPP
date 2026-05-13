include(FetchContent)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nanobench)

add_executable(wsvt_bench bench_core.cpp)
target_link_libraries(wsvt_bench PRIVATE wsvt_core nanobench)
