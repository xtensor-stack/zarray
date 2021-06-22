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

TEST_SUITE_BEGIN("zarray");

namespace xt{

  template <class T>
    void check_xarray_data_type(const std::string& data_type)
    {
        auto a = xarray<T>();
        zarray z(a);
        EXPECT_EQ(z.get_metadata()["data_type"], data_type);
    }

    TEST(zarray, constructor)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> ra = {{2., 2.}, {3., 4.}};
        zarray da(a);
        da.get_array<double>()(0, 0) = 2.;

        EXPECT_EQ(a, ra);
    }

    TEST(zarray, initializer_list)
    {
        zarray za = {1., 2., 3., 4.};
        xarray<double> xa = {1., 2., 3., 4.};
        EXPECT_EQ(za.get_array<double>(), xa);

        zarray zb = {{1., 2.}, {3., 4.}};
        xarray<double> xb = {{1., 2.}, {3., 4.}};
        EXPECT_EQ(zb.get_array<double>(), xb);

        zarray zc = {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}};
        xarray<double> xc = {{{1., 2.}, {3., 4.}}, {{5., 6.}, {7., 8.}}};
        EXPECT_EQ(zc.get_array<double>(), xc);
    }

    TEST(zarray, print)
    {
        zarray zb = {{1., 2.}, {3., 4.}};
        std::ostringstream out;
        out << zb;
        std::string res = out.str();
        std::string expected = "{{ 1.,  2.},\n { 3.,  4.}}";
        EXPECT_EQ(expected, res);
    }

    TEST(zarray, copy_constructor)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> b = a;
        xarray<double> res1 = {{2., 2.}, {3., 4.}};
        xarray<double> res2 = a;

        // da holds a reference on a
        zarray da(a);
        zarray da2(da);
        da.get_array<double>()(0, 0) = 2.;
        EXPECT_EQ(a, res1);
        EXPECT_EQ(da2.get_array<double>(), res1);

        // db holds b
        zarray db = xt::xarray<double>(b);
        zarray db2(db);
        db.get_array<double>()(0, 0) = 2.;
        EXPECT_EQ(b, res2);
        EXPECT_EQ(db2.get_array<double>(), res2);
    }

    TEST(zarray, assign_operator)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> b = {{2., 2.}, {3., 4.}};
        xarray<double> c = a;

        zarray da(a);
        zarray db(b);
        da = db;
        EXPECT_EQ(a, b);

        zarray dc = xt::xarray<double>(c);
        dc = db;
        EXPECT_EQ(dc.get_array<double>(), b);
        EXPECT_NE(c, b);

        xarray<int> d;
        xarray<int> e = {0, 1};
        zarray id(d);
        zarray ie(e);
        id = ie;
        EXPECT_EQ(d, e);
    }

    TEST(zarray, move_constructor)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> b = a;
        xarray<double> res1 = {{2., 2.}, {3., 4.}};

        // da holds a reference on a
        zarray da(a);
        da.get_array<double>()(0, 0) = 2.;
        zarray da2(std::move(da));
        EXPECT_EQ(da2.get_array<double>(), res1);

        // db holds b
        zarray db = xt::xarray<double>(b);
        db.get_array<double>()(0, 0) = 2.;
        zarray db2(std::move(db));
        EXPECT_EQ(db2.get_array<double>(), res1);
    }

    TEST(zarray, move_assign)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> b = {{2., 2.}, {3., 4.}};
        xarray<double> c = a;

        zarray da(a);
        zarray db(b);

        db = std::move(da);
        EXPECT_EQ(db.get_array<double>(), c);
    }

    TEST(zarray, extended_copy)
    {
        zdispatcher_t<detail::plus, 2>::init();
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> b = {{3., 1.}, {2., 5.}};
        xarray<double> res = {{4., 3.}, {5., 9.}};

        zarray za(a);
        zarray zb(b);
        zarray zsum = za + zb;
        EXPECT_EQ(zsum.get_array<double>(), res);
    }

    TEST(zarray, extended_assign)
    {
        zdispatcher_t<detail::plus, 2>::init();
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> b = {{3., 1.}, {2., 5.}};
        xarray<double> c = {{1., 1.}, {0., 0.}};
        xarray<double> res = {{4., 3.}, {5., 9.}};

        zarray za(a);
        zarray zb(b);
        zarray zc1 = xarray<double>(c);
        zc1 = za + zb;
        EXPECT_EQ(zc1.get_array<double>(), res);
        EXPECT_NE(c, res);

        zarray zc2(c);
        zc2 = za + zb;
        EXPECT_EQ(c, res);
    }

    TEST(zarray, shape)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        zarray za(a);

        std::size_t size = za.dimension();
        EXPECT_EQ(size, 2u);

        auto shape = za.shape();
        EXPECT_EQ(shape, a.shape());
    }

    TEST(zarray, reshape)
    {
        xarray<double> a = {1., 2., 3., 4.};
        zarray za(a);

        dynamic_shape<size_t> sh = {2u, 2u};
        za.resize(sh);
        EXPECT_EQ(a.shape(), sh);
    }

    TEST(zarray, resize)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        zarray za(a);

        dynamic_shape<size_t> sh = {3u, 4u, 2u};
        za.resize(sh);
        EXPECT_EQ(a.shape(), sh);
    }

    TEST(zarray, broadcast_shape)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        zarray za(a);

        dynamic_shape<size_t> sh1 = {1u, 1u};
        dynamic_shape<size_t> res1 = {2u, 2u};

        za.broadcast_shape(sh1);
        EXPECT_EQ(sh1, res1);

        dynamic_shape<size_t> sh2 = {3u, 1u, 2u};
        dynamic_shape<size_t> res2 = {3u, 2u, 2u};

        za.broadcast_shape(sh2);
        EXPECT_EQ(sh2, res2);
    }

    TEST(zarray, assign_xarray)
    {
        auto a1 = xarray<int>();
        auto a2 = xarray<int>({0, 1});
        zarray z1(a1);
        z1 = a2;
        EXPECT_EQ(a1, a2);
    }

    TEST(zarray, noalias_assign)
    {
        auto a1 = xarray<int>();
        auto a2 = xarray<int>({0, 1});
        zarray z1(a1), z2(a2);
        noalias(z1) = z2;
        EXPECT_EQ(a1, a2);
    }

    TEST(zarray, noalias_assign_xarray)
    {
        auto a1 = xarray<int>();
        auto a2 = xarray<int>({0, 1});
        zarray z1(a1);
        noalias(z1) = a2;
        EXPECT_EQ(a1, a2);
    }

    TEST(zarray, data_type)
    {
        std::string s = (xtl::endianness() == xtl::endian::little_endian) ? "<" : ">";
        check_xarray_data_type<bool>("bool");
        check_xarray_data_type<uint8_t>("u1");
        check_xarray_data_type<uint16_t>(s + "u2");
        check_xarray_data_type<uint32_t>(s + "u4");
        check_xarray_data_type<uint64_t>(s + "u8");
        check_xarray_data_type<int8_t>("i1");
        check_xarray_data_type<int16_t>(s + "i2");
        check_xarray_data_type<int32_t>(s + "i4");
        check_xarray_data_type<int64_t>(s + "i8");
        check_xarray_data_type<xtl::half_float>(s + "f2");
        check_xarray_data_type<float>(s + "f4");
        check_xarray_data_type<double>(s + "f8");
    }
}

TEST_SUITE_END(); // end of testsuite gm