cmake_minimum_required(VERSION 2.8)

find_package(PkgConfig)
pkg_check_modules(EIGEN REQUIRED eigen3)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})

add_executable(main main.cc)
target_link_libraries(main gpio)

add_subdirectory(gpio)
