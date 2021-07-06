/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "test_common.hpp"

#include <zarray/zarray.hpp>
#include <xtl/xplatform.hpp>
#include <xtl/xhalf_float.hpp>


TEST_SUITE_BEGIN("zchunked_array");

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

    TEST(zchunked_array, iterator)
    {
        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);
        zarray za(a);

        auto it = a.chunk_cbegin();
        auto it_end = a.chunk_cend();
        auto zit = za.as_chunked_array().chunk_begin();
        
        while(it != it_end)
        {
            const auto& tmp = zit.get_xchunked_iterator<decltype(it)>();
            EXPECT_EQ(tmp, it);
            ++it, ++zit;
        }
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

    TEST(zchunked_array, assign_operator)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);
        auto b = xarray<double>::from_shape(shape);
        b.fill(2.);
        
        zarray za(a);
        zarray zb(b);
        za = zb;
        EXPECT_EQ(a, b);
    }

    TEST(zchunked_array, assign_xarray)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);
        auto b = xarray<double>::from_shape(shape);
        b.fill(2.);
        
        zarray za(a);
        za = b;
        EXPECT_EQ(a, b);
    }

    TEST(zchunked_array, noalias_assign)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);
        auto b = xarray<double>::from_shape(shape);
        b.fill(2.);
        
        zarray za(a);
        zarray zb(b);
        noalias(za) = zb;
        EXPECT_EQ(a, b);
    }

    TEST(zchunked_array, no_noalias_assign)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);
        auto b = xarray<double>::from_shape(shape);
        b.fill(2.);

        zarray za(a);
        zarray zb(b);
        za = zb;
        EXPECT_EQ(a, b);
    }

    TEST(zchunked_array, noalias_assign_xarray)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);
        auto b = xarray<double>::from_shape(shape);
        b.fill(2.);
        
        zarray za(a);
        noalias(za) = b;
        EXPECT_EQ(a, b);
    }

    TEST(zchunked_array, bug20)
    {
        std::vector<size_t> shape = {4, 4};
        std::vector<size_t> chunk_shape = {2, 2};
        auto a1 = chunked_array<int>(shape, chunk_shape);
        auto a2 = chunked_array<int>(shape, chunk_shape) = arange(4 * 4).reshape({4, 4});
        zarray z1(a1);
        z1 = a2;

        EXPECT_EQ(a1, a2);
    }
}

TEST_SUITE_END(); 