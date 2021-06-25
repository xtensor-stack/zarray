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
#include "xtensor-io/xchunk_store_manager.hpp"
#include "xtensor-io/xio_binary.hpp"
#include "xtensor-io/xio_disk_handler.hpp"
#include "xtensor/xview.hpp"
#include "test_init.hpp"

TEST_SUITE_BEGIN("zview");

namespace xt
{
    namespace fs = ghc::filesystem;

    TEST(zview, strided_view)
    {
        initialize_dispatchers();

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xstrided_slice_vector sv1({xt::all(), 1});
        xstrided_slice_vector sv2({1});
        xarray<double> expected1 = xt::strided_view(a, sv1);
        xarray<double> expected2 = xt::strided_view(expected1, sv2);

        zarray za(a);

        zarray zres1 = strided_view(za, sv1);
        zarray zres2 = strided_view(zres1, sv2);

        EXPECT_EQ(zres1.get_array<double>(), expected1);
        EXPECT_EQ(zres2.get_array<double>(), expected2);
    }

    TEST(zview, chunked_strided_view)
    {
        initialize_dispatchers();

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

        zarray zres = strided_view(za, sv);

        EXPECT_EQ(zres.get_array<double>(), expected);
    }

    TEST(zview, assign_strided_view)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xstrided_slice_vector sv({all(), 1});
        xarray<double> b = {{1., 5.}, {3., 6.}};
        xarray<double> expected = {{1., 5.}, {3., 6.}};

        zarray za(a), zb(b);
        strided_view(za, sv) = strided_view(zb, sv);
        EXPECT_EQ(za.get_array<double>(), expected);

        xarray<double> expected2 = {{2., 5.}, {6., 6.}};
        xstrided_slice_vector sv2({all(), 0});
        strided_view(za, sv2) = strided_view(zb, sv2) + strided_view(zb, sv2);
        EXPECT_EQ(za.get_array<double>(), expected2);

        xarray<double> c = {7., 8.};
        xarray<double> expected3 = {{7., 8.}, {6., 6.}};
        xstrided_slice_vector sv3({0, all()});
        strided_view(za, sv3) = c;
        EXPECT_EQ(za.get_array<double>(), expected3);
    }

    TEST(zview, assign_chunked_strided_view)
    {
        std::vector<size_t> shape = {4, 4};
        std::vector<size_t> chunk_shape = {2, 2};
        std::string chunk_dir = "chunk_dir2";
        fs::create_directory(chunk_dir);
        double init_value = 123.456;
        auto ca = chunked_file_array<double, xio_disk_handler<xio_binary_config>>(shape, chunk_shape, chunk_dir, init_value);

        xarray<double> a = {{1., 2.}, {3., 4.}};
        xstrided_slice_vector sv({range(2, 4), range(2, 4)});
        xarray<double> expected = ones<double>({4, 4}) * init_value;
        strided_view(expected, sv) = a;

        zarray za(ca);
        strided_view(za, sv) = a;

        EXPECT_EQ(za.get_array<double>(), expected);
    }
}

TEST_SUITE_END(); 