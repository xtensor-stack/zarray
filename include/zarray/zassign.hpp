/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZASSIGN_HPP
#define XTENSOR_ZASSIGN_HPP

#include "xtensor/xassign.hpp"
#include "zarray_impl.hpp"

namespace xt
{
    struct zassign_args
    {
        zassign_args();
        bool trivial_broadcast;
    };

    inline zassign_args::zassign_args()
        : trivial_broadcast(false)
    {
    }

    namespace detail
    {
        template <class Tag>
        struct zexpression_assigner
        {
            // Both E1 and E2 are zarray expressions
            template <class E1, class E2>
            static void assign_data(E1& e1, const E2& e2, const zassign_args& args)
            {
                e2.assign_to(e1.get_implementation(), args);
            }
        };

        template <>
        struct zexpression_assigner<xtensor_expression_tag>
        {
            // E1 is a zarray_expression, E2 is an xtensor_expression
            template <class E1, class E2>
            static void assign_data(E1& e1, const E2& e2, const zassign_args&)
            {
                using value_type = typename E2::value_type;
                using array_type = ztyped_array<value_type>;
                array_type& ar = dynamic_cast<array_type&>(e1.get_implementation());
                if (ar.is_array())
                {
                    xt::noalias(ar.get_array()) = e2;
                }
                else
                {
                    xarray<value_type> tmp(e2);
                    ar.assign(std::move(tmp));
                }
            }
        };
    }

    template <>
    class xexpression_assigner<zarray_expression_tag>
    {
    public:

        template <class E1, class E2>
        static void assign_xexpression(xexpression<E1>& e1, const xexpression<E2>& e2)
        {
            E1& lhs = e1.derived_cast();
            const E2& rhs = e2.derived_cast();
            std::size_t size = rhs.dimension();
            auto shape = uninitialized_shape<dynamic_shape<std::size_t>>(size);
            zassign_args args;
            args.trivial_broadcast = rhs.broadcast_shape(shape, true);
            lhs.resize(std::move(shape));
            detail::zexpression_assigner<xexpression_tag_t<E2>>::assign_data(lhs, rhs, args);
        }
    };

    /******************************
     * zassign_wrapped_expression *
     ******************************/

    template <class E1, class E2>
    inline void zassign_wrapped_expression(xexpression<E1>& e1, const xexpression<E2>& e2, const zassign_args& args)
    {
        assign_data(e1, e2, args.trivial_broadcast);
    }

    template <class T, class E2>
    inline void zassign_wrapped_expression(ztyped_array<T>& lhs, const xexpression<E2>& rhs, const zassign_args& args)
    {
        if (lhs.is_array())
        {
            zassign_wrapped_expression(lhs.get_array(), rhs, args);
        }
        else
        {
            xarray<T> tmp(rhs);
            lhs.assign(std::move(tmp));
        }
    }
}

#endif
