############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 3.1)

find_package(doctest            REQUIRED)
find_package(Threads)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(zarray-test)

    enable_testing()

    find_package(zarray REQUIRED CONFIG)
    set(ZARRAY_INCLUDE_DIR ${zarray_INCLUDE_DIRS})
endif ()

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting tests build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif()

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

include(set_compiler_flag.cmake)

if(CPP17)
  # User requested C++17, but compiler might not oblige.
  set_compiler_flag(
    _cxx_std_flag CXX
    "-std=c++17"  # this should work with GNU, Intel, PGI
    "/std:c++17"  # this should work with MSVC
  )
  if(_cxx_std_flag)
    message(STATUS "Building with C++17")
  endif()
elseif(CPP20)
  # User requested C++17, but compiler might not oblige.
  set_compiler_flag(
    _cxx_std_flag CXX
    "-std=c++2a"  # this should work with GNU, Intel, PGI
    "/std:c++2a"  # this should work with MSVC
  )
  if(_cxx_std_flag)
    message(STATUS "Building with C++20")
  endif()
else()
  set_compiler_flag(
    _cxx_std_flag CXX REQUIRED
    "-std=c++14"  # this should work with GNU, Intel, PGI
    "/std:c++14"  # this should work with MSVC
  )
  message(STATUS "Building with C++14")
endif()

if(NOT _cxx_std_flag)
  message(FATAL_ERROR "zarray needs a C++14-compliant compiler.")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
  CHECK_CXX_COMPILER_FLAG(-march=native arch_native_supported)
  if(arch_native_supported AND NOT CMAKE_CXX_FLAGS MATCHES "-march")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable -ftemplate-backtrace-limit=0")
  if (XTENSOR_DISABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /MP /bigobj")
  # set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO)
  # add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  # add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
  if (XTENSOR_DISABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c-")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  if(NOT WIN32)
    CHECK_CXX_COMPILER_FLAG(-march=native arch_native_supported)
    if(arch_native_supported AND NOT CMAKE_CXX_FLAGS MATCHES "-march")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} -Wunused-parameter -Wextra -Wreorder -Wconversion -Wsign-conversion")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable")
    if (XTENSOR_DISABLE_EXCEPTIONS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
    endif()
  else() # We are using clang-cl
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_cxx_std_flag} /bigobj")
    set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
    if (XTENSOR_DISABLE_EXCEPTIONS)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c-")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
    endif()
  endif()
else()
  message(FATAL_ERROR  ${CMAKE_CXX_COMPILER_ID} "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

set(ZARRAY_TESTS
    test_init.cpp
    test_zarray.cpp
    test_zchunked_array.cpp
    test_zfunction.cpp
    test_zreducer_options.cpp
    test_zreducer.cpp
    test_zreducer_norms.cpp
    test_zreducer_statistics.cpp
    test_zreducer_minmax.cpp
    test_zview.cpp
    test_zexpression_tree.cpp)

foreach(filename IN LISTS ZARRAY_TESTS)
    string(REPLACE ".cpp" "" targetname ${filename})
    add_executable(${targetname} main.cpp ${targetname} ${filename} test_init.cpp)
    if(ZARRAY_USE_XSIMD)
        target_compile_definitions(${targetname}
                                   PRIVATE
                                   XTENSOR_USE_XSIMD)
        target_link_libraries(${targetname} PRIVATE xsimd)
    endif()

    target_include_directories(${targetname} PRIVATE ${ZARRAY_INCLUDE_DIR})
    target_link_libraries(${targetname} PRIVATE zarray doctest::doctest ${CMAKE_THREAD_LIBS_INIT})

    target_compile_options(${targetname} PRIVATE -g -O0)

    add_custom_target(
        x${targetname}
        COMMAND ${targetname}
        DEPENDS ${targetname} ${filename} ${ZARRAY_HEADERS})
    add_test(NAME ${targetname} COMMAND ${targetname})
endforeach()

add_executable(test_zarray_lib main.cpp  ${ZARRAY_TESTS})
if(ZARRAY_USE_XSIMD)
    target_compile_definitions(test_zarray_lib
                               PRIVATE
                               XTENSOR_USE_XSIMD)
    target_link_libraries(test_zarray_lib PRIVATE xsimd)
endif()

target_include_directories(test_zarray_lib PRIVATE ${ZARRAY_INCLUDE_DIR})
target_link_libraries(test_zarray_lib PRIVATE zarray doctest::doctest ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(xtest COMMAND test_zarray_lib DEPENDS test_zarray_lib)
add_test(NAME xtest COMMAND test_zarray_lib)


