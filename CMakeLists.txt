
cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER g++-11)

list(APPEND CMAKE_PREFIX_PATH "~/libtorch")

project(selfplay)

include_directories(
    includes
)

add_executable(
    selfplay 
    # tests/testmain.cc
    # tests/multi_play.cc
    sources/selfplay_main.cc
    # tests/indicator_test.cc
    sources/mcts/tree.cc
    sources/mcts/node.cc
    sources/mcts/noise.cc
    sources/gomoku/board.cc
    sources/gomoku/evaluator.cc
    sources/gomoku/eval_queue.cc
    sources/gomoku/server.cc
    sources/gomoku/logger.cc
    sources/gomoku/utils.cc
)


set(CAFFE2_USE_CUDNN 1)
find_package(Torch REQUIRED)
list(APPEND TORCH_LIBRARIES "/usr/local/cuda/extras/CUPTI/lib64/libcupti.so")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
target_link_libraries(selfplay "${TORCH_LIBRARIES}")

find_package(Boost 1.30 COMPONENTS program_options REQUIRED)
target_include_directories(selfplay PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(selfplay ${Boost_LIBRARIES})

add_subdirectory(libraries/fmt)
find_package(fmt)
target_link_libraries(selfplay fmt::fmt)

add_subdirectory(libraries/indicators)
find_package(indicators)
target_link_libraries(selfplay indicators::indicators)


set_target_properties(selfplay
PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)


