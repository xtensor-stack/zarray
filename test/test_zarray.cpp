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
    template <class T>
    void check_xarray_data_type(const std::string& data_type)
    {
        auto a = xarray<T>();
        zarray z(a);
        EXPECT_EQ(z.attrs()["data_type"], data_type);
    }

    class xattrs
    {
    public:
        const nlohmann::json& attrs();
        void set_attrs(const nlohmann::json& attrs);

    private:
        nlohmann::json m_attrs;
    };

    inline const nlohmann::json& xattrs::attrs()
    {
        return m_attrs;
    }

    inline void xattrs::set_attrs(const nlohmann::json& attrs)
    {
        m_attrs = attrs;
    }

    TEST(zarray, constructor)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> ra = {{2., 2.}, {3., 4.}};
        zarray da(a);
        da.get_array<double>()(0, 0) = 2.;

        EXPECT_EQ(a, ra);
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

    // TODO: move to dedicated test file
    TEST(zarray, chunked_array)
    {
        using shape_type =  zarray::shape_type;
        shape_type shape = {10, 10, 10};
        shape_type chunk_shape = {2, 3, 4};
        auto a = chunked_array<double>(shape, chunk_shape);

        zarray za(a);
        shape_type res = za.as_chunked_array().chunk_shape();
        EXPECT_EQ(res, chunk_shape);
    }

    TEST(zarray, zarray_assign_xarray)
    {
        auto a1 = xarray<int>();
        auto a2 = xarray<int>({0, 1});
        zarray z1(a1);
        z1 = a2;
        EXPECT_EQ(a1, a2);
    }

    TEST(zarray, noalias_zarray_assign_xarray)
    {
        auto a1 = xarray<int>();
        auto a2 = xarray<int>({0, 1});
        zarray z1(a1);
        noalias(z1) = a2;
        EXPECT_EQ(a1, a2);
    }

    TEST(zarray, xarray_data_type)
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

    TEST(zarray, chunked_array_attrs)
    {
        using shape_type =  zarray::shape_type;
        shape_type shape = {4, 4};
        shape_type chunk_shape = {2, 2};
        auto a = chunked_array<double, XTENSOR_DEFAULT_LAYOUT, xattrs>(shape, chunk_shape);
        nlohmann::json attrs;
        attrs["foo"] = "bar";
        a.set_attrs(attrs);
        zarray z(a);
        EXPECT_EQ(z.attrs()["foo"], "bar");
    }

    TEST(zarray, chunked_array_noattrs)
    {
        std::string s = (xtl::endianness() == xtl::endian::little_endian) ? "<" : ">";
        using shape_type =  zarray::shape_type;
        shape_type shape = {4, 4};
        shape_type chunk_shape = {2, 2};
        auto a = chunked_array<double>(shape, chunk_shape);
        zarray z(a);
        EXPECT_EQ(z.attrs()["data_type"], s + "f8");
    }
}
