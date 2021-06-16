############################################################################
# Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          #
# Copyright (c) QuantStack                                                 #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

find_package(Threads)
    find_package(doctest            REQUIRED)

set(ZARRAY_TESTS
    test_zarray.cpp
    test_zfunction.cpp
    test_init.cpp
    test_zview.cpp
    test_zchunked_array.cpp
)


add_executable( doctest
    main.cpp
    ${ZARRAY_TESTS}
)


target_include_directories(doctest PRIVATE ${ZARRAY_INCLUDE_DIR})
target_link_libraries(doctest PRIVATE zarray doctest::doctest nlohmann_json::nlohmann_json)

add_custom_target(xdoctest COMMAND doctest DEPENDS doctest)
add_test(NAME xdoctest COMMAND doctest)


