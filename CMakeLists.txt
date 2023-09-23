
cmake_minimum_required(VERSION 3.20)
set(CMAKE_CXX_COMPILER g++-11)

list(APPEND CMAKE_PREFIX_PATH "~/libtorch")

project(selfplay)

include_directories(
    includes
)

add_executable(
    selfplay 
    # tests/testmain.cc
    tests/testplay.cc
    sources/mcts/tree.cc
    sources/mcts/node.cc
    sources/mcts/noise.cc
    sources/gomoku/board.cc
    sources/gomoku/evaluator.cc
    sources/gomoku/eval_queue.cc
    sources/gomoku/selfplay.cc
    sources/gomoku/logger.cc
    sources/gomoku/utils.cc
)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")
target_link_libraries(selfplay "${TORCH_LIBRARIES}")

add_subdirectory(libraries/fmt)
find_package(fmt)
target_link_libraries(selfplay fmt::fmt)

set_target_properties(selfplay
PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)


