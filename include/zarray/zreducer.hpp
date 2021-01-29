/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZREDUCER_HPP
#define XTENSOR_ZREDUCER_HPP

#include <tuple>
#include <utility>

#include "zdispatcher.hpp"
#include "xtensor/xreducer.hpp"

namespace xt
{
    class zarray;

    template <class CT, class Func, class AX, class OX>
    class zreducer : public xexpression<zreducer<CT>>
    {
    public:

        using expression_tag = zarray_expression_tag;

        using self_type = zreducer<CT>;
        using functor_type = detail::xreducer_dummy_functor;
        using shape_type = dynamic_shape<std::size_t>;

        template <class FuncA, class CTA, class AXA, class OXA>
        zreducer(FuncA&& func, CTA&& e, AXA&& axes, OXA&& options) noexcept;

        std::size_t dimension() const;
        void broadcast_shape(shape_type& shape, bool reuse_cache = false) const;

        std::unique_ptr<zarray_impl> allocate_result() const;
        std::size_t get_result_type_index() const;
        zarray_impl& assign_to(zarray_impl& res) const;

    private:

        using dispatcher_type = zdispatcher_t<functor_type, 1, 1>;

        std::size_t compute_dimension() const;

        std::size_t get_result_type_index_impl() const;

        zarray_impl& assign_to_impl(zarray_impl& res) const;

        CT m_e;
        mutable shape_type m_cache_shape;
        mutable bool m_cache_initialized;
        //i have to create some kind of m_args
        //        Func m_func;
        //AX m_axes;
        //OX m_options;
        //are all of these args gonna be inside of my m_args?
        detail::strided_view_args<detail::no_adj_strides_policy> m_args;
    };

    /************************
     * zreducer implementation *
     ************************/
    
    template <class CT, class Func, class AX, class OX>
    template <class FuncA, class CTA, class AXA, class OXA>
    inline zreducer<CT>::zreducer(FuncA&& func, CTA&& e, AXA&& axes, OXA&& options)noexcept
    //do i need forward here
        : m_e(std::forward<CTA>(e))
        : m_func(std::forward<FuncA>(func))
        : m_axes(std::forward<AXA>(axes))
        : m_options(std::forward<OXA>(options))
    {
        auto shape = e.shape();
        auto offset = e.get_implementation().get_offset();
        auto layout = e.layout();
        m_args.fill_args(shape, strides, offset, layout, slices);
    }

    template <class CT, class Func, class AX, class OX>
    inline std::size_t zreducer<CT>::dimension() const
    {
        return typename OX::keep_dims() ? m_e.dimension() : m_e.dimension() - m_axes.size();
    }

    template <class CT>
    inline void zreducer<CT>::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        if (reuse_cache && m_cache_initialized)
        {
            std::copy(m_cache_shape.cbegin(), m_cache_shape.cend(), shape.begin());
        }
        else
        {
            //maybe instead of calling m_e.shape() like this I should use m_args.shape
            xt::broadcast_shape(m_e.shape(), shape);
        }
    }
}

#endif
