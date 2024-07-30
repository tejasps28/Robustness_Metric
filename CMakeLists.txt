cmake_minimum_required(VERSION 2.8.3)
project(robustness)

FIND_PACKAGE(PythonInterp REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED)
FIND_PACKAGE(Boost COMPONENTS python REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O2 -DFMT_HEADER_ONLY")

find_package(Eigen3 REQUIRED)

include_directories(
  include
  include/basalt/thirdparty
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
)

add_executable(basalt_test src/basalt_test.cpp)
target_link_libraries(basalt_test Eigen3::Eigen armadillo fmt)

add_executable(eval_robustness src/eval_robustness.cpp)
target_link_libraries(eval_robustness Eigen3::Eigen armadillo fmt ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(eval_robustness_batch src/eval_robustness_batch.cpp)
target_link_libraries(eval_robustness_batch Eigen3::Eigen armadillo fmt ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

# Python module
PYTHON_ADD_MODULE(RobustMetricLib src/eval_robustness.cpp)
target_link_libraries(RobustMetricLib ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} Eigen3::Eigen armadillo fmt)
set_target_properties(RobustMetricLib PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/script)

