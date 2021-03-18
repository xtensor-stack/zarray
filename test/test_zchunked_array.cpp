/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <gtest/gtest.h>
#include <zarray/zarray.hpp>
#include <xtl/xplatform.hpp>
#include <xtl/xhalf_float.hpp>

namespace xt
{
    TEST(zchunked_array, chunked_array)
    {
        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);

        zarray za(a);
        shape_type res = za.as_chunked_array().chunk_shape();
        EXPECT_EQ(res, chunk_shape);
    }

    TEST(zchunked_array, custom_metadata)
    {
        using shape_type =  zarray::shape_type;
        shape_type shape = {4, 4};
        shape_type chunk_shape = {2, 2};
        auto a = chunked_array<double, XTENSOR_DEFAULT_LAYOUT>(shape, chunk_shape);
        zarray z(a);
        nlohmann::json metadata;
        metadata["foo"] = "bar";
        z.set_metadata(metadata);
        EXPECT_EQ(z.get_metadata()["foo"], "bar");
    }
}
