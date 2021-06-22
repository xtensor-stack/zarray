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

TEST_SUITE_BEGIN("zreducers");

namespace xt
{

    auto params_amin_amax(){
        return std::make_tuple(
            std::make_tuple([](auto && ... args){return amin(std::forward<decltype(args)>(args)...);},
                            [](auto && ... args){return zt::amin(std::forward<decltype(args)>(args)...);}),
            std::make_tuple([](auto && ... args){return amax(std::forward<decltype(args)>(args)...);},
                            [](auto && ... args){return zt::amax(std::forward<decltype(args)>(args)...);})
            );
    }
    HETEROGEN_PARAMETRIZED_DEFINE(REDUCERS_AMIN_AMAX)
    {
        using value_type = double;
        auto params = get_param<TypeParam>(params_amin_amax());
        auto && xred = std::get<0>(params);
        auto && zred = std::get<1>(params);

        auto a = xarray<value_type>::from_shape({4,6,8});
        std::iota(a.begin(), a.end(), value_type(0));
        auto za = zarray(a);

        const std::size_t axes[1] = {1};
        auto reducer = zred(za, axes);
        auto should_reducer = xred(a, axes);
        auto should_res = xt::eval(should_reducer);
        auto res_shape = should_res.shape();

        // check the dimensions
        EXPECT_EQ(reducer.dimension(), should_reducer.dimension());
        EXPECT_EQ(reducer.shape(), should_reducer.shape());

        zarray zres = reducer;
        EXPECT_TRUE(zres.can_get_array<value_type>());
        auto & res = zres.get_array<value_type>();
        EXPECT_EQ(res, should_res);

    }
    HETEROGEN_PARAMETRIZED_TEST_APPLY(REDUCERS_AMIN_AMAX, params_amin_amax);
}

TEST_SUITE_END();
