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
        xstrided_slice_vector slices;
        size_t chunk_index;
    };

    inline zassign_args::zassign_args()
        : trivial_broadcast(false)
        , slices()
        , chunk_index(0u)
    {
    }

    namespace detail
    {
        template <class E1, class E2, class F>
        void run_chunked_assign_loop(E1 & e1, const E2& e2, zassign_args& args, F f)
        {
            const zchunked_array& arr = e1.as_chunked_array();
            size_t grid_size = arr.grid_size();
            for (size_t i = 0; i < grid_size; ++i)
            {
                args.slices = arr.get_slice_vector(i);
                args.chunk_index = i;
                f(e1, e2, args);
            }
        }

        template <class Tag>
        struct zexpression_assigner
        {
            // Both E1 and E2 are zarray expressions
            template <class E1, class E2>
            static void assign_data(E1& e1, const E2& e2, zassign_args& args)
            {
                if (e1.get_implementation().is_chunked())
                {
                    auto l = [](E1& e1, const E2& e2, zassign_args& args)
                    {
                        e2.assign_to(e1.get_implementation(), args);
                    };
                    run_chunked_assign_loop(e1, e2, args, l);
                }
                else
                {
                    e2.assign_to(e1.get_implementation(), args);
                }
            }
        };

        template <>
        struct zexpression_assigner<xtensor_expression_tag>
        {
            // E1 is a zarray_expression, E2 is an xtensor_expression
            template <class E1, class E2>
            static void assign_data(E1& e1, const E2& e2, zassign_args& /*args*/)
            {
                using value_type = typename E2::value_type;
                zarray_impl& impl = e1.get_implementation();
                if (impl.is_array())
                {
                    using array_type = ztyped_array<value_type>;
                    array_type& ar = static_cast<array_type&>(impl);
                    xt::noalias(ar.get_array()) = e2;
                }
                else if (impl.is_chunked())
                {
                    using array_type = ztyped_chunked_array<value_type>;
                    array_type& ar = static_cast<array_type&>(impl);
                    size_t grid_size = ar.grid_size();
                    for (size_t i = 0; i < grid_size; ++i)
                    {
                        ar.assign_chunk(strided_view(e2, ar.get_slice_vector(i)), i);
                    }
                }
                else
                {
                    using array_type = ztyped_expression_wrapper<value_type>;
                    array_type& ar = static_cast<array_type&>(impl);
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
        else if (!args.slices.empty())
        {
            xarray<T> tmp(rhs);
            auto& chunked_lhs = static_cast<ztyped_chunked_array<T>&>(lhs);
            chunked_lhs.assign_chunk(std::move(tmp), args.chunk_index); 
        }
        else
        {
            xarray<T> tmp(rhs);
            auto& expr_lhs = static_cast<ztyped_expression_wrapper<T>&>(lhs);
            expr_lhs.assign(std::move(tmp));
        }
    }
}

#endif
