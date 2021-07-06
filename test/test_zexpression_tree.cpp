#include "test_common.hpp"

#include <zarray/zinit.hpp>
#include <zarray/zarray.hpp>

#include <algorithm>

#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>


TEST_SUITE_BEGIN("zexpression_tree");
namespace xt
{

    TEST_CASE("count_temporaries")
    {

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();
        //init_zsystem<void>();

        auto x0 = xarray<float>::from_shape({2,2});
        auto x1 = xarray<float>::from_shape({2,2});
        auto x2 = xarray<float>::from_shape({2,2});
        auto x3 = xarray<float>::from_shape({2,2});
        auto x4 = xarray<float>::from_shape({2,2});
        auto x5 = xarray<float>::from_shape({2,2});
        auto x6 = xarray<float>::from_shape({2,2});
        auto x7 = xarray<float>::from_shape({2,2});

        zarray z0(x0);
        zarray z1(x1);
        zarray z2(x2);
        zarray z3(x3);
        zarray z4(x4);
        zarray z5(x5);
        zarray z6(x6);
        zarray z7(x7);

        SUBCASE("array_leaves")
        {
            auto func = ((z0 + z1) + (z2 + z3)) + ((z4 + z5) + (z6 + z7));
        
            SUBCASE("heterogen assign")
            {
                auto res = xarray<double>::from_shape({2,2});
                zarray zres(res);
                detail::zarray_temporary_pool temporary_pool(zres.get_implementation());

                zassign_args assign_args;
                func.assign_to(temporary_pool, assign_args);

                CHECK_EQ(temporary_pool.size(), 3);
            }
            SUBCASE("homogen assign")
            {
                auto res = xarray<float>::from_shape({2,2});
                zarray zres(res);
                detail::zarray_temporary_pool temporary_pool(zres.get_implementation());

                zassign_args assign_args;
                func.assign_to(temporary_pool, assign_args);

                CHECK_EQ(temporary_pool.size(), 2);
            }
        }
        SUBCASE("reducer_leaves")
        {
            auto f = [](const auto & a ){return zt::sum(a, {0});};
            auto func = ((f(z0) + f(z1)) + (f(z2) + f(z3))) + ((f(z4) + f(z5)) + (f(z6) + f(z7)));

            auto res = xarray<float>::from_shape({2,2});
            zarray zres(res);
            detail::zarray_temporary_pool temporary_pool(zres.get_implementation());

            zassign_args assign_args;
            func.assign_to(temporary_pool, assign_args);
            CHECK_EQ(temporary_pool.size(), 3);
        }
        SUBCASE("chain")
        {
            auto func = z0 + z1 + z2 + z3 + z4 + z5 + z6 + z7;
            auto res = xarray<float>::from_shape({2,2});
            zarray zres(res);
            detail::zarray_temporary_pool temporary_pool(zres.get_implementation());

            zassign_args assign_args;
            func.assign_to(temporary_pool, assign_args);
            CHECK_EQ(temporary_pool.size(), 0);
        }
    }

    TEST_CASE("non_trivial_expression_tree")
    {

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();
        //init_zsystem<void>();

        auto x0 = xarray<double>::from_shape({2,2});
        auto x1 = xarray<double>::from_shape({2,2});
        auto x2 = xarray<double>::from_shape({2,2});
        auto x3 = xarray<double>::from_shape({2,2});
        auto x4 = xarray<double>::from_shape({2,2});
        auto x5 = xarray<double>::from_shape({2,2});
        auto x6 = xarray<double>::from_shape({2,2});
        auto x7 = xarray<double>::from_shape({2,2});
        auto res = xarray<double>::from_shape({2,2});


        std::iota(x0.begin(), x0.end(), 0);
        std::iota(x1.begin(), x1.end(), 4);
        std::iota(x2.begin(), x2.end(), 2*4);
        std::iota(x3.begin(), x3.end(), 3*4);
        std::iota(x4.begin(), x4.end(), 4*4);
        std::iota(x5.begin(), x5.end(), 5*4);
        std::iota(x6.begin(), x6.end(), 6*4);
        std::iota(x7.begin(), x7.end(), 7*4);

        zarray z0(x0);
        zarray z1(x1);
        zarray z2(x2);
        zarray z3(x3);
        zarray z4(x4);
        zarray z5(x5);
        zarray z6(x6);
        zarray z7(x7);
        zarray zres(res);

        SUBCASE("chain")
        {
            zres      = z0 + z1 + z2 + z3 + z4 + z5 + z6 + z7;
            auto xres = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            CHECK_EQ(res, xres);
        }

        SUBCASE("tree")
        {
            zres      = ((z0 + z1) + (z2 + z3)) + ((z4 + z5) + (z6 + z7));
            auto xres = ((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7));
            CHECK_EQ(res, xres);
        }

        SUBCASE("tree_with_reducers")
        {
            zres      = ((z0 + zt::sum(z1, {0})) + zt::sum(z2 + z3, {0})) + ((z4 + z5) + (z6 + zt::sum(z7)));
            auto xres = ((x0 +     sum(x1, {0})) +     sum(x2 + x3, {0})) + ((x4 + x5) + (x6 +     sum(x7)));
            CHECK_EQ(res, xres);
        }
    }

    TEST_CASE("assign_to")
    {

        zdispatcher_t<detail::plus, 2>::init();
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        //init_zsystem<void>();

        auto x0 = xarray<float>::from_shape({2,2});
        auto x1 = xarray<float>::from_shape({2,2});

        zarray z0(x0);
        zarray z1(x1);

        auto res = xarray<float>::from_shape({2,2});
        zarray zres(res);

        zassign_args assign_args;
        auto func = ( zt::sum(z0,{0}) +  zt::sum(z1,{0}))  + (z0 + z1);
        func.assign_to(zres.get_implementation(), assign_args);
    }


    TEST_CASE("partial_chunked_tree")
    {

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        using value_type = double;
        using shape_type =  zarray::shape_type;

        shape_type shape = {8, 9, 10};
        shape_type chunk_shape = {2, 3, 4};


        auto x0 = xarray<double>::from_shape(shape);
        auto x1 = chunked_array<double>(shape, chunk_shape);
        auto x2 = xarray<double>::from_shape(shape);
        auto x3 = chunked_array<double>(shape, chunk_shape);

        zarray z0(x0);
        zarray z1(x1);
        zarray z2(x2);
        zarray z3(x3);

        auto should_res = (x0 + x1) + value_type(2) + (x2 + x3);
        auto func =       (z0 + z1) + value_type(2) + (z2 + z3);
        
        SUBCASE("assign_to_chunked")
        {
            auto res = chunked_array<double>(shape, chunk_shape);
            zarray zres(res);
            zres =  func;
            CHECK_EQ(res, should_res);
        }
        SUBCASE("assign_to_non_chunked")
        {
            auto res = xarray<double>::from_shape(shape);
            zarray zres(res);
            zres =  func;
            CHECK_EQ(res, should_res);
        }
    }

    TEST_CASE("test_casting_order")
    {
        zdispatcher_t<detail::plus, 2>::init();  
        zdispatcher_t<detail::multiplies, 2>::init(); 
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        xarray<double> x0 = xt::ones<double>({2}) * 0.1; 
        xarray<double> x1 = xt::ones<double>({2}) * 0.1;
        xarray<double> x2 = xt::ones<double>({2}) * 10.0;
        xarray<int64_t> xres = xarray<int64_t>::from_shape({2});

        zarray z0(x0);
        zarray z1(x1);
        zarray z2(x2);
        zarray zres(xres);

        SUBCASE("direct")
        {
            noalias(zres) = (z0 + z1) * z2;
            CHECK_EQ(xres(0), 2);
        }
        SUBCASE("indirect")
        {
            auto func = (z0 + z1) * z2;
            zassign_args assign_args;
            func.assign_to(zres.get_implementation(), assign_args);
            CHECK_EQ(xres(0), 2);
        }
    }
}
TEST_SUITE_END();

