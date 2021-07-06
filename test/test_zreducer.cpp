/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include "test_common.hpp"

#include "zarray/zarray.hpp"
#include "zarray/zreducer.hpp"
#include "zarray/zreducers.hpp"

TEST_SUITE_BEGIN("zreducers");

namespace xt
{
    using axes_vec = xt::dynamic_shape<std::size_t>;

    TEST(zreducer, factory_closure_types)
    {
        {
            xarray<float> a = {{1, 2}, {3, 4}};
            zarray za(a);
            using should_reducer_type = zreducer<zsum_zreducer_functor, const zarray&>;

            auto reducer = make_zreducer<zsum_zreducer_functor>(za, zreducer_options{});
            using reducer_type = std::decay_t<decltype(reducer)>;

            static_assert(std::is_same<should_reducer_type, reducer_type>::value, "types are not the same");
        }
        {

            using should_reducer_type = zreducer<zsum_zreducer_functor, zarray>;

            auto reducer = make_zreducer<zsum_zreducer_functor>(zarray{1.0,2.0}, zreducer_options{});
            using reducer_type = std::decay_t<decltype(reducer)>;

            static_assert(std::is_same<should_reducer_type, reducer_type>::value, "types are not the same");
        }
    }

    // tests where axis and options are given
    auto axis_and_options_params()
    {
        return std::make_tuple(
            std::make_tuple(axes_vec{},         keep_dims),
            std::make_tuple(axes_vec{},         xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0},        keep_dims),
            std::make_tuple(axes_vec{0},        xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{1},        keep_dims),
            std::make_tuple(axes_vec{1},        xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{2},        keep_dims),
            std::make_tuple(axes_vec{2},        xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0,1},      keep_dims),
            std::make_tuple(axes_vec{0,1},      xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0,2},      keep_dims),
            std::make_tuple(axes_vec{0,2},      xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{1,2},      keep_dims),
            std::make_tuple(axes_vec{1,2},      xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0,1,2},    keep_dims),
            std::make_tuple(axes_vec{0,1,2},    xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{},         initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{},         initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0},        initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{0},        initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{1},        initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{1},        initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{2},        initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{2},        initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0,1},      initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{0,1},      initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0,2},      initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{0,2},      initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{1,2},      initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{1,2},      initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{0,1,2},    initial(1.0f)|keep_dims),
            std::make_tuple(axes_vec{0,1,2},    initial(1.0f)|xt::evaluation_strategy::lazy),
            std::make_tuple(axes_vec{},         initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{},         initial(1.0f)|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0},        initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0},        initial(1.0f)|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{1},        initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{1},        initial(1.0f)|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{2},        initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{2},        initial(1.0f)|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0,1},      initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0,1},      initial(1.0f)|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0,2},      initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0,2},      initial(1.0f)|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{1,2},      initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{1,2},      initial(1.0f)|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0,1,2},    initial(1.0f)|keep_dims|xt::evaluation_strategy::immediate),
            std::make_tuple(axes_vec{0,1,2},    initial(1.0f)|xt::evaluation_strategy::immediate)
        );
    }
    HETEROGEN_PARAMETRIZED_DEFINE(SHAPE_TEST_WITH_AXIS)
    {
        auto params = get_param<TypeParam>(axis_and_options_params());
        auto && axes = std::get<0>(params);
        auto && options = std::get<1>(params);

        auto a = xarray<float>::from_shape({4,6,8});
        std::iota(a.begin(), a.end(), 0);
        auto za = zarray(a);

        auto reducer = zt::sum(za, axes, options);
        auto should_reducer = sum(a, axes , options);
        auto should_res = xt::eval(should_reducer);

        dynamic_shape<std::size_t> res_shape(should_res.shape().begin(), should_res.shape().end());

        // check the dimensions
        EXPECT_EQ(reducer.dimension(), should_reducer.dimension());
        EXPECT_EQ(reducer.shape(), res_shape);

        // check the values on assignment
        {
            zarray zres = reducer;
            EXPECT_TRUE(zres.can_get_array<float>());
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
        {
            auto res = xarray<float>::from_shape(should_res.shape());
            zarray zres(res);
            zres = reducer;
            EXPECT_EQ(res, should_res);
        }
        {
            auto res = xarray<float>::from_shape(should_res.shape());
            zarray zres(res);
            noalias(zres) = reducer;
            EXPECT_EQ(res, should_res);
        }
        {
            zarray zres(reducer);
            EXPECT_TRUE(zres.can_get_array<float>());
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
        if(!axes.empty()) // skip this due to an xtensor bug
        {
            {
                auto chunk_shape =  dynamic_shape<std::size_t> (res_shape.size(), 2);
                auto res = chunked_array<float>(res_shape, chunk_shape);
                zarray zres(res);
                zres = reducer;
                EXPECT_EQ(res, should_res);
            }
            {
                auto chunk_shape =  dynamic_shape<std::size_t> (res_shape.size(), 2);
                auto res = chunked_array<float>(res_shape, chunk_shape);
                zarray zres(res);
                noalias(zres) = reducer;
                EXPECT_EQ(res, should_res);
            }
        }
        {
            zarray zres(std::move(reducer));
            EXPECT_TRUE(zres.can_get_array<float>());
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
    }
    HETEROGEN_PARAMETRIZED_TEST_APPLY(SHAPE_TEST_WITH_AXIS, axis_and_options_params);

    // tests where only options are given
    auto option_params(){
        return std::make_tuple(
            xt::evaluation_strategy::immediate,
            xt::evaluation_strategy::lazy,
            initial(1.0f)|xt::evaluation_strategy::immediate,
            initial(1.0f)|xt::evaluation_strategy::lazy,
            keep_dims|xt::evaluation_strategy::immediate,
            keep_dims|xt::evaluation_strategy::lazy,
            keep_dims|initial(1.0f)|xt::evaluation_strategy::immediate,
            keep_dims|initial(1.0f)|xt::evaluation_strategy::lazy
        );
    }
    HETEROGEN_PARAMETRIZED_DEFINE(SHAPE_TEST_NO_AXIS)
    {
        auto options = get_param<TypeParam>(option_params());

        auto a = xarray<float>::from_shape({2,3,4});
        std::iota(a.begin(), a.end(), 0);
        auto za = zarray(a);

        auto reducer = zt::sum(za, options);
        auto should_reducer = sum(a, options);
        auto should_res = xt::eval(should_reducer);
        dynamic_shape<std::size_t> should_shape(should_res.shape().begin(), should_res.shape().end());

        EXPECT_EQ(reducer.dimension(), should_reducer.dimension());
        EXPECT_EQ(reducer.shape(), should_shape);

        // check the values on assignment
        {
            zarray zres = reducer;
            EXPECT_TRUE(zres.can_get_array<float>());
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
        {
            auto res = xarray<float>::from_shape(should_res.shape());
            zarray zres(res);
            zres = reducer;
            EXPECT_EQ(res, should_res);
        }
        {
            auto res = xarray<float>::from_shape(should_res.shape());
            zarray zres(res);
            noalias(zres) = reducer;
            EXPECT_EQ(res, should_res);
        }
        {
            zarray zres(reducer);
            EXPECT_TRUE(zres.can_get_array<float>());
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
        {
            zarray zres(std::move(reducer));
            EXPECT_TRUE(zres.can_get_array<float>());
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
    }
    HETEROGEN_PARAMETRIZED_TEST_APPLY(SHAPE_TEST_NO_AXIS, option_params);

    TEST(zreducer, nested_reducers)
    {
        {
            xarray<float> a = {{1, 2}, {3, 4}};
            zarray za(a);

            //zarray zres = sum(za, zreducer_options({0}, false, false));
            auto reducer_inner = zt::sum(za,           {0}, xt::evaluation_strategy::lazy);
            auto reducer_outer = zt::sum(reducer_inner,{0}, xt::evaluation_strategy::lazy);
            zarray zres = reducer_outer;

            // do the same with plain xtensor
            auto should_inner = xt::sum(a, {0}, xt::evaluation_strategy::lazy);
            auto should_res = xt::eval(xt::sum(should_inner, {0}, xt::evaluation_strategy::lazy));

            // cast
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
        {
            xarray<float> a = {{1, 2}, {3, 4}};
            zarray za(a);

            zarray zres = zt::sum( zt::sum(za,{0}),{0});;

            // do the same with plain xtensor
            auto should_res = xt::eval(xt::sum( xt::sum(a, {0}), {0}));

            // cast
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
    }


    TEST(zreducer, zfunction)
    {
        {
            xarray<float> a = {{1, 2}, {3, 4}};
            xarray<float> b = {{2, 4}, {6, 8}};
            zarray za(a);
            zarray zb(b);

            auto func = za + zb;
            auto reducer = zt::sum(func,{0});
            zarray zres = reducer;

            // do the same with plain xtensor
            auto should_func =  a + b;
            auto should_res = xt::eval(xt::sum(should_func, {0}));

            // cast
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
        {
            xarray<float> a = {{1, 2}, {3, 4}};
            xarray<float> b = {{2, 4}, {6, 8}};
            zarray za(a);
            zarray zb(b);
            zarray zres = zt::sum( za + zb,{0});

            // do the same with plain xtensor
            auto should_res = xt::eval(xt::sum(a + b, {0}));

            // cast
            auto & res = zres.get_array<float>();
            EXPECT_EQ(res, should_res);
        }
    }
}

TEST_SUITE_END();