#include "test_common.hpp"

#include <zarray/zarray.hpp>
#include <xtl/xplatform.hpp>
#include <xtl/xhalf_float.hpp>

TEST_SUITE_BEGIN("zfunction");
namespace xt
{
  TEST(zfunction, shape)
    {
        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> b = {0.5, 1.5};
        zarray za(a);
        zarray zb(b);

        using zfunction_type = zfunction<detail::plus, const zarray&, const zarray&>;
        zfunction_type f(zplus(), za, zb);

        auto dim = f.dimension();
        EXPECT_EQ(dim, a.dimension());

        auto shape = f.shape();
        EXPECT_EQ(shape, a.shape());

        decltype(shape) shape2 = {1u, 2u};
        f.broadcast_shape(shape2);
        EXPECT_EQ(shape2, a.shape());
    }

    TEST(zfunction, dispatching)
    {
        using dispatcher_type = zdispatcher_t<math::exp_fun, 1>;
        dispatcher_type::init();

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> expa = {{std::exp(0.5), std::exp(1.5)}, {std::exp(2.5), std::exp(3.5)}};
        auto res = xarray<double>::from_shape({2, 2});
        zarray za(a);
        zarray zres(res);

        zassign_args args;
        dispatcher_type::dispatch(za.get_implementation(), zres.get_implementation(), args);

        EXPECT_EQ(expa, res);
    }

    TEST(zfunction, assign_to)
    {
        using exp_dispatcher_type = zdispatcher_t<math::exp_fun, 1>;
        exp_dispatcher_type::init();

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        using nested_zfunction_type = zfunction<math::exp_fun, const zarray&>;
        using zfunction_type = zfunction<detail::plus, const zarray&, nested_zfunction_type>;

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> b = {{-0.2, 2.4}, {1.3, 4.7}};
        auto res = xarray<double>::from_shape({2, 2});

        zarray za(a);
        zarray zb(b);
        zarray zres(res);

        zfunction_type f(zplus(), za, nested_zfunction_type(zexp(), zb));
        zassign_args args;
        f.assign_to(zres.get_implementation(), args);

        auto expected = xarray<double>::from_shape({2, 2});
        std::transform(a.cbegin(), a.cend(), b.cbegin(), expected.begin(),
                       [](const double& lhs, const double& rhs) { return lhs + std::exp(rhs); });

        EXPECT_TRUE(all(isclose(res, expected)));

        size_t res_index = f.get_result_type_index();
        EXPECT_EQ(res_index, ztyped_array<double>::get_class_static_index());
    }

    TEST(zfunction, math_operator)
    {
        using exp_dispatcher_type = zdispatcher_t<math::exp_fun, 1>;
        exp_dispatcher_type::init();

        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        xarray<double> a = {{0.5, 1.5}, {2.5, 3.5}};
        xarray<double> b = {{-0.2, 2.4}, {1.3, 4.7}};
        auto res = xarray<double>::from_shape({2, 2});

        zarray za(a);
        zarray zb(b);
        zarray zres(res);

        auto f = za + xt::exp(zb);
        zassign_args args;
        f.assign_to(zres.get_implementation(), args);

        auto expected = xarray<double>::from_shape({2, 2});
        std::transform(a.cbegin(), a.cend(), b.cbegin(), expected.begin(),
                       [](const double& lhs, const double& rhs) { return lhs + std::exp(rhs); });

        EXPECT_TRUE(all(isclose(res, expected)));
    }

    TEST(zfunction, scalar)
    {
        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> res;

        zarray za(a);
        zarray zres(res);

        auto f = za + 2.;
        using expected_type = zfunction<detail::plus, const zarray&, zscalar_wrapper<xscalar<double>>>;
        EXPECT_TRUE((std::is_same<decltype(f), expected_type>::value));
        
        zres = f;
        xarray<double> expected = {{3., 4.}, {5., 6.}};
        EXPECT_EQ(zres.get_array<double>(), expected);
    }

    TEST(zfunction, broadcasting)
    {
        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> b = {1., 2.};
        xarray<double> expected = a + b;

        zarray za(a);
        zarray zb(b);
        zarray zres = za + zb;

        EXPECT_EQ(zres.get_array<double>(), expected);
    }

    TEST(zfunction, math_operator_extended)
    {
        using add_dispatcher_type = zdispatcher_t<detail::plus, 2>;
        add_dispatcher_type::init();

        xarray<double> a = {{1, 1}, {1, 1}};
        xarray<double> b = {{2, 2}, {2, 2}};
        xarray<double> c = {{3, 3}, {3, 3}};
        xarray<double> d = {{4, 4}, {4, 4}};

        auto res = xarray<double>::from_shape({2, 2});

        zarray za(a);
        zarray zb(b);
        zarray zc(c);
        zarray zd(d);
        zarray zres(res);

        auto f = ((za+zb) + (zc+zd)) + zd;

        zassign_args args;

        f.assign_to(zres.get_implementation(), args);

        auto expected = xt::eval((a+b) + (c+d) + d);
        EXPECT_EQ(res, expected);
    }
}
TEST_SUITE_END();