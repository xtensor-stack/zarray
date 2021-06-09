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
#include "zarray/zreducer.hpp"
#include "zarray/zmath.hpp"

namespace xt
{

    using axes_vec = std::vector<std::size_t>;

    TEST(zreducer_options, options)
    {
        // {
        //     zreducer_options opts({});
        //     EXPECT_FALSE(opts.keep_dims());
        //     EXPECT_TRUE(opts.is_lazy());
        //     EXPECT_EQ(opts.axes().size(), 0);
        //     EXPECT_TRUE(opts.axes().empty());
        //     EXPECT_FALSE(opts.has_initial_value());
        // }

        {
            zreducer_options opts(axes_vec{0}, xt::evaluation_strategy::immediate|xt::initial(1.0));
            EXPECT_FALSE(opts.keep_dims());
            EXPECT_FALSE(opts.is_lazy());
            EXPECT_EQ(opts.axes().size(),1);
            EXPECT_TRUE(opts.has_initial_value());
            EXPECT_TRUE(opts.can_get_inital_value<double>());
        }
        {
            zreducer_options opts(axes_vec{0}, xt::evaluation_strategy::immediate);
            EXPECT_FALSE(opts.keep_dims());
            EXPECT_FALSE(opts.is_lazy());
            EXPECT_FALSE(opts.has_initial_value());
        }
        {
            zreducer_options opts(axes_vec{0}, xt::keep_dims|  xt::evaluation_strategy::immediate);
            EXPECT_TRUE(opts.keep_dims());
            EXPECT_FALSE(opts.is_lazy());
            EXPECT_FALSE(opts.has_initial_value());
        }
        {
            zreducer_options opts(axes_vec{0}, xt::keep_dims);
            EXPECT_TRUE(opts.keep_dims());
            EXPECT_TRUE(opts.is_lazy());
            EXPECT_FALSE(opts.has_initial_value());
        }
        {
            zreducer_options opts(axes_vec{0}, xt::keep_dims |  xt::evaluation_strategy::lazy);
            EXPECT_TRUE(opts.keep_dims());
            EXPECT_TRUE(opts.is_lazy());
            EXPECT_FALSE(opts.has_initial_value());
        }
        {
            zreducer_options opts(axes_vec{0}, xt::evaluation_strategy::lazy);
            EXPECT_FALSE(opts.keep_dims());
            EXPECT_FALSE(opts.has_initial_value());
        }
    }

}

