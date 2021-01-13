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
#include "xtensor-io/xchunk_store_manager.hpp"
#include "xtensor-io/xio_binary.hpp"
#include "xtensor-io/xio_disk_handler.hpp"
#include "xtensor/xview.hpp"

namespace xt
{
    namespace fs = ghc::filesystem;

    TEST(zview, assign_to)
    {
        using dispatcher_type = zdispatcher_t<detail::xview_dummy_functor, 1, 1>;
        dispatcher_type::init();

        using zfunction_type = zview<const zarray&>;

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xt::xstrided_slice_vector sv({xt::all(), 1});
        xarray<double> expected = xt::strided_view(a, sv);
        auto res = xarray<double>::from_shape({2});

        zarray za(a);
        zarray zres(res);

        zfunction_type f(za, sv);
        f.assign_to(zres.get_implementation());

        EXPECT_EQ(res, expected);

        size_t res_index = f.get_result_type_index();
        EXPECT_EQ(res_index, ztyped_array<double>::get_class_static_index());
    }

    TEST(zview, strided_view)
    {
        init_zsystem();

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xstrided_slice_vector sv({xt::all(), 1});
        xarray<double> expected = xt::strided_view(a, sv);

        zarray za(a);

        zarray zres = make_strided_view(za, sv);

        EXPECT_EQ(zres.get_array<double>(), expected);
    }

    TEST(zview, chunked_strided_view)
    {
        init_zsystem();

        std::vector<size_t> shape = {4, 4};
        std::vector<size_t> chunk_shape = {2, 2};
        std::string chunk_dir = "chunk_dir";
        fs::create_directory(chunk_dir);
        double init_value = 123.456;
        auto ca = chunked_file_array<double, xio_disk_handler<xio_binary_config>>(shape, chunk_shape, chunk_dir, init_value);
        xarray<double> a = {{1., 2.}, {3., 4.}};
        view(ca, range(0, 2), range(0, 2)) = a;
        xt::xstrided_slice_vector sv({xt::all(), 1});
        xarray<double> expected = {2, 4, init_value, init_value};

        zarray za(ca);

        zarray zres = make_strided_view(za, sv);

        EXPECT_EQ(zres.get_array<double>(), expected);
    }
}
