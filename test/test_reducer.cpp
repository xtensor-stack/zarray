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
    TEST(reducer, mean)
    {
        xarray<double> a = {{1., 2.}, {3., 4.}};
        xarray<double> expected = xt::mean(a);

        zarray za(a);
        zarray zres = mean(za);

        EXPECT_EQ(zres.get_array<double>(), expected);
    }
}
