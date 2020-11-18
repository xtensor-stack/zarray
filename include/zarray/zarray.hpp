/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZARRAY_HPP
#define XTENSOR_ZARRAY_HPP

#include <memory>

#include <xtl/xmultimethods.hpp>

#include "xtensor/xarray.hpp"
#include "zarray_impl.hpp"
#include "zassign.hpp"
#include "zfunction.hpp"
#include "zmath.hpp"

namespace xt
{

    /**********
     * zarray *
     **********/

    class zarray;

    template <>
    struct xcontainer_inner_types<zarray>
    {
        using temporary_type = zarray;
    };

    class zarray : public xcontainer_semantic<zarray>
    {
    public:

        using expression_tag = zarray_expression_tag;
        using semantic_base = xcontainer_semantic<zarray>;
        using implementation_ptr = std::unique_ptr<zarray_impl>;
        using shape_type = zarray_impl::shape_type;

        zarray() = default;
        ~zarray() = default;

        zarray(implementation_ptr&& impl);
        zarray& operator=(implementation_ptr&& impl);

        zarray(const zarray& rhs);
        zarray& operator=(const zarray& rhs);

        zarray(zarray&& rhs);
        zarray& operator=(zarray&& rhs);

        template <class E>
        zarray(const xexpression<E>& e);

        template <class E>
        zarray(xexpression<E>& e);

        template <class E>
        zarray(xexpression<E>&& e);
        
        template <class E>
        zarray& operator=(const xexpression<E>&);

        void swap(zarray& rhs);

        zarray_impl& get_implementation();
        const zarray_impl& get_implementation() const;

        template <class T>
        xarray<T>& get_array();

        template <class T>
        const xarray<T>& get_array() const;

        std::size_t dimension() const;
        const shape_type& shape() const;
        void resize(const shape_type& shape);
        void resize(shape_type&& shape);
        void broadcast_shape(shape_type& shape, bool reuse_cache = false) const;

        const zchunked_array& as_chunked_array() const;

    private:

        template <class E>
        void init_implementation(E&& e, xtensor_expression_tag);

        template <class E>
        void init_implementation(const xexpression<E>& e, zarray_expression_tag);

        implementation_ptr p_impl;
    };

    /*************************
     * zarray implementation *
     *************************/

    template <class E>
    inline void zarray::init_implementation(E&& e, xtensor_expression_tag)
    {
        p_impl = implementation_ptr(detail::build_zarray(std::forward<E>(e)));
    }

    template <class E>
    inline void zarray::init_implementation(const xexpression<E>& e, zarray_expression_tag)
    {
        p_impl = e.derived_cast().allocate_result();
        semantic_base::assign(e);
    }

    inline zarray::zarray(implementation_ptr&& impl)
        : p_impl(std::move(impl))
    {
    }

    inline zarray& zarray::operator=(implementation_ptr&& impl)
    {
        p_impl = std::move(impl);
        return *this;
    }

    inline zarray::zarray(const zarray& rhs)
        : p_impl(rhs.p_impl->clone())
    {
    }

    inline zarray& zarray::operator=(const zarray& rhs)
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::dispatch(*(rhs.p_impl), *p_impl);
        return *this;
    }

    inline zarray::zarray(zarray&& rhs)
        : p_impl(std::move(rhs.p_impl))
    {
    }

    inline zarray& zarray::operator=(zarray&& rhs)
    {
        zdispatcher_t<detail::xmove_dummy_functor, 1>::dispatch(*(rhs.p_impl), *p_impl);
        return *this;
    }

    template <class E>
    inline zarray::zarray(const xexpression<E>& e)
    {
        init_implementation(e.derived_cast(), extension::get_expression_tag_t<std::decay_t<E>>());
    }

    template <class E>
    inline zarray::zarray(xexpression<E>& e)
    {
        init_implementation(e.derived_cast(), extension::get_expression_tag_t<std::decay_t<E>>());
    }

    template <class E>
    inline zarray::zarray(xexpression<E>&& e)
    {
        init_implementation(std::move(e).derived_cast(), extension::get_expression_tag_t<std::decay_t<E>>());
    }

    template <class E>
    inline zarray& zarray::operator=(const xexpression<E>& e)
    {
        return semantic_base::operator=(e);
    }
    
    inline void zarray::swap(zarray& rhs)
    {
        std::swap(p_impl, rhs.p_impl);
    }

    inline zarray_impl& zarray::get_implementation()
    {
        return *p_impl;
    }

    inline const zarray_impl& zarray::get_implementation() const
    {
        return *p_impl;
    }

    template <class T>
    inline xarray<T>& zarray::get_array()
    {
        return dynamic_cast<ztyped_array<T>*>(p_impl.get())->get_array();
    }

    template <class T>
    inline const xarray<T>& zarray::get_array() const
    {
        return dynamic_cast<const ztyped_array<T>*>(p_impl.get())->get_array();
    }

    inline std::size_t zarray::dimension() const
    {
        return p_impl->dimension();
    }

    inline auto zarray::shape() const -> const shape_type&
    {
        return p_impl->shape();
    }

    inline void zarray::resize(const shape_type& shape)
    {
        p_impl->resize(shape);
    }

    inline void zarray::resize(shape_type&& shape)
    {
        p_impl->resize(std::move(shape));
    }

    inline void zarray::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        p_impl->broadcast_shape(shape, reuse_cache);
    }

    inline const zchunked_array& zarray::as_chunked_array() const
    {
        return dynamic_cast<const zchunked_array&>(*(p_impl.get()));
    }
}

#endif
