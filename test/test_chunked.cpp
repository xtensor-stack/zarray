/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "gtest/gtest.h"
#include "zarray/zarray.hpp"

namespace xt
{
    TEST(zarray, chunk_shape)
    {
        using shape_type = zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);

        zarray za(a);
        shape_type res = za.as_chunked_array().chunk_shape();
        EXPECT_EQ(res, chunk_shape);
    }

    TEST(zarray, assign_chunked_array)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        std::vector<size_t> shape = {4, 4};
        std::vector<size_t> chunk_shape = {2, 2};
        auto a1 = chunked_array<int>(shape, chunk_shape);
        auto a2 = chunked_array<int>(shape, chunk_shape) = arange(4 * 4).reshape({4, 4});
        zarray z1(a1);
        z1 = a2;
        EXPECT_EQ(a1, a2);
    }
}
