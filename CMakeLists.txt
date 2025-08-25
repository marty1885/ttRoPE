cmake_minimum_required(VERSION 3.15)

project(ttRoPE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

find_package(PNG REQUIRED)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CPM)

if(CMAKE_VERSION VERSION_LESS 3.25)
    add_subdirectory(dependencies EXCLUDE_FROM_ALL)
else()
    add_subdirectory(dependencies EXCLUDE_FROM_ALL SYSTEM)
endif()

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()

add_library(ttmetal IMPORTED INTERFACE)
target_include_directories(ttmetal INTERFACE
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/api/
    $ENV{TT_METAL_HOME}/tt_metal/include
    $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon/api
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_stl/
)

target_link_directories(ttmetal INTERFACE
    $ENV{TT_METAL_HOME}/build/tt_metal
)

target_link_libraries(ttmetal INTERFACE
    fmt
    yaml-cpp
    Boost::core
    Boost::container
    tt_metal
)

add_executable(cpu main.cpp)
add_executable(ttrope ttrope/main.cpp)
target_link_libraries(ttrope PRIVATE ttmetal)
target_precompile_headers(ttrope PRIVATE ttrope/pch.hpp)
