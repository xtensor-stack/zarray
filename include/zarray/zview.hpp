/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZVIEW_HPP
#define XTENSOR_ZVIEW_HPP

#include <tuple>
#include <utility>

#include "zdispatcher.hpp"
#include "xtensor/xstrided_view.hpp"

namespace xt
{
    class zarray;

    template <class CT>
    class zview : public xexpression<zview<CT>>
    {
    public:

        using expression_tag = zarray_expression_tag;

        using self_type = zview<CT>;
        using functor_type = detail::xview_dummy_functor;
        using shape_type = dynamic_shape<std::size_t>;

        template <class CTA>
        zview(CTA&& e, xstrided_slice_vector& slices) noexcept;

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
        xstrided_slice_vector& m_slices;
        mutable shape_type m_cache_shape;
        mutable bool m_cache_initialized;
        detail::strided_view_args<detail::no_adj_strides_policy> m_args;
    };

    /************************
     * zview implementation *
     ************************/
    
    template <class CT>
    template <class CTA>
    inline zview<CT>::zview(CTA&& e, xstrided_slice_vector& slices) noexcept
        : m_e(std::forward<CTA>(e))
        , m_slices(slices)
        , m_cache_shape()
        , m_cache_initialized(false)
    {
        auto shape = e.shape();
        auto strides = e.get_strides();
        auto offset = e.get_implementation().get_offset();
        auto layout = e.layout();
        m_args.fill_args(shape, strides, offset, layout, slices);
    }

    template <class CT>
    inline std::size_t zview<CT>::dimension() const
    {
        return m_args.new_shape.size();
    }

    template <class CT>
    inline void zview<CT>::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        if (reuse_cache && m_cache_initialized)
        {
            std::copy(m_cache_shape.cbegin(), m_cache_shape.cend(), shape.begin());
        }
        else
        {
            xt::broadcast_shape(m_args.new_shape, shape);
        }
    }

    template <class CT>
    inline std::unique_ptr<zarray_impl> zview<CT>::allocate_result() const
    {
        std::size_t idx = get_result_type_index();
        return std::unique_ptr<zarray_impl>(zarray_impl_register::get(idx).clone());
    }

    template <class CT>
    inline std::size_t zview<CT>::get_result_type_index() const
    {
        return get_result_type_index_impl();
    }

    template <class CT>
    inline zarray_impl& zview<CT>::assign_to(zarray_impl& res) const
    {
        return assign_to_impl(res);
    }

    template <class CT>
    inline std::size_t zview<CT>::compute_dimension() const
    {
        return m_e.dimension();
    }

    template <class CT>
    std::size_t zview<CT>::get_result_type_index_impl() const
    {
        return dispatcher_type::get_type_index(
                zarray_impl_register::get(
                    detail::get_result_type_index(m_e)
                )
               );
    }

    template <class CT>
    inline zarray_impl& zview<CT>::assign_to_impl(zarray_impl& res) const
    {
        dispatcher_type::dispatch(detail::get_array_impl(m_e, res), res, m_args);
        return res;
    }
}

#endif
