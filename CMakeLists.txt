############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)
project(zarray)

set(ZARRAY_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Versionning
# ===========

file(STRINGS "${ZARRAY_INCLUDE_DIR}/zarray/zarray_config.hpp" zarray_version_defines
    REGEX "#define ZARRAY_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${zarray_version_defines})
    if(ver MATCHES "#define ZARRAY_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(ZARRAY_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
    ${ZARRAY_VERSION_MAJOR}.${ZARRAY_VERSION_MINOR}.${ZARRAY_VERSION_PATCH})
message(STATUS "Building zarray v${${PROJECT_NAME}_VERSION}")

# Dependencies
# ============

set(xtensor_REQUIRED_VERSION 0.23.0)
if(TARGET xtensor)
    set(xtensor_VERSION
        ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtensor_VERSION} VERSION_GREATER_EQUAL ${xtensor_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but
        requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()

find_package(nlohmann_json 3.2.0 REQUIRED)

# Optional dependencies
# =====================

OPTION(ZARRAY_USE_XSIMD "simd acceleration for zarray" OFF)

if(ZARRAY_USE_XSIMD)
    set(xsimd_REQUIRED_VERSION 7.4.4)
    if(TARGET xsimd)
        set(xsimd_VERSION ${XSIMD_VERSION_MAJOR}.${XSIMD_VERSION_MINOR}.${XSIMD_VERSION_PATCH})
        # Note: This is not SEMVER compatible comparison
        if( NOT ${xsimd_VERSION} VERSION_GREATER_EQUAL ${xsimd_REQUIRED_VERSION})
            message(ERROR "Mismatch xsimd versions. Found '${xsimd_VERSION}' but requires: '${xsimd_REQUIRED_VERSION}'")
        else()
            message(STATUS "Found xsimd v${xsimd_VERSION}")
        endif()
    else()
        find_package(xsimd ${xsimd_REQUIRED_VERSION} REQUIRED)
        message(STATUS "Found xsimd: ${xsimd_INCLUDE_DIRS}/xsimd")
    endif()
endif()

# Build
# =====

set(ZARRAY_HEADERS
    ${ZARRAY_INCLUDE_DIR}/zarray/zarray.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zarray_config.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zarray_impl.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zarray_wrapper.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zassign.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zchunked_iterator.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zchunked_wrapper.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zdispatcher.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zdispatching_types.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zexpression_wrapper.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zfunction.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zmath.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zscalar_wrapper.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zwrappers.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zfunctors.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zarray_impl_register.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zmpl.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zarray_temporary_pool.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zinit.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zreducers.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zreducer.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zreducer_options.hpp
    ${ZARRAY_INCLUDE_DIR}/zarray/zarray_zarray.hpp
)

add_library(zarray INTERFACE)

target_include_directories(zarray INTERFACE
    $<BUILD_INTERFACE:${ZARRAY_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>)

target_compile_features(zarray INTERFACE cxx_std_14)

target_link_libraries(zarray INTERFACE xtensor)
target_link_libraries(zarray INTERFACE nlohmann_json::nlohmann_json)

OPTION(BUILD_TESTS "zarray test suite" OFF)
OPTION(DOWNLOAD_GTEST "build gtest from downloaded sources" OFF)
OPTION(CPP17 "enables C++17" OFF)
OPTION(CPP20 "enables C++20 (experimental)" OFF)
OPTION(DISABLE_MSVC_ITERATOR_CHECK "Disable the MVSC iterator check" ON)


if(MSVC AND DISABLE_MSVC_ITERATOR_CHECK)
    add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)
endif()

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Installation
# ============

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS zarray
        EXPORT ${PROJECT_NAME}-targets)

# Makes the project importable from the build directory
export(EXPORT ${PROJECT_NAME}-targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

   install(FILES ${ZARRAY_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zarray)

    set(ZARRAY_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
    STRING "install path for zarrayConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
                              "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                              INSTALL_DESTINATION ${ZARRAY_CMAKECONFIG_INSTALL_DIR})

# zarray is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from zarrayConfigVersion.cmake so that an zarrayConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_ZARRAY_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                 VERSION ${${PROJECT_NAME}_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_ZARRAY_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
              DESTINATION ${ZARRAY_CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}-targets
        FILE ${PROJECT_NAME}Targets.cmake
        DESTINATION ${ZARRAY_CMAKECONFIG_INSTALL_DIR})

