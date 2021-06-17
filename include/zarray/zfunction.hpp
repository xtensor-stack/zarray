/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZFUNCTION_HPP
#define XTENSOR_ZFUNCTION_HPP

#include <tuple>
#include <utility>

#include "zdispatcher.hpp"
#include "zarray_impl_register.hpp"

namespace xt
{
    template <class F, class... CT>
    class zfunction : public xexpression<zfunction<F, CT...>>
    {
    public:

        using expression_tag = zarray_expression_tag;

        using self_type = zfunction<F, CT...>;
        using tuple_type = std::tuple<CT...>;
        using functor_type = F;
        using shape_type = dynamic_shape<std::size_t>;

        template <class Func, class... CTA, class U = std::enable_if_t<!std::is_base_of<std::decay_t<Func>, self_type>::value>>
        zfunction(Func&& f, CTA&&... e) noexcept;

        std::size_t dimension() const;
        const shape_type& shape() const;
        bool broadcast_shape(shape_type& shape, bool reuse_cache = false) const;

        std::unique_ptr<zarray_impl> allocate_result() const;
        std::size_t get_result_type_index() const;
        zarray_impl& assign_to(zarray_impl& res, const zassign_args& args) const;

    private:

        using dispatcher_type = zdispatcher_t<F, sizeof...(CT)>;

        std::size_t compute_dimension() const;

        template <std::size_t... I>
        std::size_t get_result_type_index_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        zarray_impl& assign_to_impl(std::index_sequence<I...>, zarray_impl& res, const zassign_args& args) const;

        struct cache
        {
            cache() : m_shape(), m_initialized(false), m_trivial_broadcast(false) {}

            shape_type m_shape;
            bool m_initialized;
            bool m_trivial_broadcast;
        };

        tuple_type m_e;
        mutable cache m_cache;
    };

    namespace detail
    {
        template <class E>
        struct zargument_type
        {
            using type = E;
        };

        template <class T>
        struct zargument_type<xscalar<T>>
        {
            using type = zscalar_wrapper<xscalar<T>>;
        };

        template <class E>
        using zargument_type_t = typename zargument_type<E>::type;

        template <class F, class... E>
        struct select_xfunction_expression<zarray_expression_tag, F, E...>
        {
            using type = zfunction<F, zargument_type_t<E>...>;
        };
    }

    /****************************
     * zfunction implementation *
     ****************************/

    class zarray;
    
    namespace detail
    {
        template <class E>
        struct zfunction_argument
        {
            static std::size_t get_index(const E& e)
            {
                return e.get_result_type_index();
            }

            static const zarray_impl& get_array_impl(const E& e, zarray_impl& res, const zassign_args& args)
            {
                return e.assign_to(res, args);
            }
        };

        template <>
        struct zfunction_argument<zarray>
        {
            template <class E>
            static std::size_t get_index(const E& e)
            {
                return e.get_implementation().get_class_index();
            }

            template <class E>
            static const zarray_impl& get_array_impl(const E& e, zarray_impl&, const zassign_args&)
            {
                return e.get_implementation();
            }
        };

        template <class CTE>
        struct zfunction_argument<zscalar_wrapper<CTE>>
        {
            static std::size_t get_index(const zscalar_wrapper<CTE>& e)
            {
                return e.get_class_index();
            }

            static const zarray_impl& get_array_impl(const zscalar_wrapper<CTE>& e, zarray_impl&, const zassign_args&)
            {
                return e;
            }
        };

        template <class E>
        inline size_t get_result_type_index(const E& e)
        {
            return zfunction_argument<E>::get_index(e);
        }

        template <class E>
        inline const zarray_impl& get_array_impl(const E& e, zarray_impl& z, const zassign_args& args)
        {
            return zfunction_argument<E>::get_array_impl(e, z, args);
        }
    }

    template <class F, class... CT>
    template <class Func, class... CTA, class U>
    inline zfunction<F, CT...>::zfunction(Func&&, CTA&&... e) noexcept
        : m_e(std::forward<CTA>(e)...)
        , m_cache()
    {
    }

    template <class F, class... CT>
    inline std::size_t zfunction<F, CT...>::dimension() const
    {
        return m_cache.m_initialized ? m_cache.m_shape.size() : compute_dimension();
    }

    template <class F, class... CT>
    inline auto zfunction<F, CT...>::shape() const -> const shape_type&
    {
        if (!m_cache.m_initialized)
        {
            m_cache.m_shape = uninitialized_shape<shape_type>(compute_dimension());
            m_cache.m_trivial_broadcast = broadcast_shape(m_cache.m_shape, false);
            m_cache.m_initialized = true;
        }
        return m_cache.m_shape;
    }

    template <class F, class... CT>
    inline bool zfunction<F, CT...>::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        if (reuse_cache && m_cache.m_initialized)
        {
            std::copy(m_cache.m_shape.cbegin(), m_cache.m_shape.cend(), shape.begin());
            return m_cache.m_trivial_broadcast;
        }
        else
        {
            auto func = [&shape](bool b, const auto& e) { return e.broadcast_shape(shape) && b; };
            return accumulate(func, true, m_e);
        }
    }
    
    template <class F, class... CT>
    inline std::unique_ptr<zarray_impl> zfunction<F, CT...>::allocate_result() const
    {
        std::size_t idx = get_result_type_index();
        return std::unique_ptr<zarray_impl>(zarray_impl_register::get(idx).clone());
    }

    template <class F, class... CT>
    inline std::size_t zfunction<F, CT...>::get_result_type_index() const
    {
        return get_result_type_index_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class... CT>
    inline zarray_impl& zfunction<F, CT...>::assign_to(zarray_impl& res, const zassign_args& args) const
    {
        return assign_to_impl(std::make_index_sequence<sizeof...(CT)>(), res, args);
    }

    template <class F, class... CT>
    inline std::size_t zfunction<F, CT...>::compute_dimension() const
    {
        auto func = [](std::size_t d, auto&& e) noexcept { return (std::max)(d, e.dimension()); };
        return accumulate(func, std::size_t(0), m_e);
    }

    template <class F, class... CT>
    template <std::size_t... I>
    std::size_t zfunction<F, CT...>::get_result_type_index_impl(std::index_sequence<I...>) const
    {
        return dispatcher_type::get_type_index(
                zarray_impl_register::get(
                    detail::get_result_type_index(std::get<I>(m_e))
                )...
               );
    }

    template <class F, class... CT>
    template <std::size_t... I>
    inline zarray_impl& zfunction<F, CT...>::assign_to_impl(std::index_sequence<I...>, zarray_impl& res, const zassign_args& args) const
    {
        dispatcher_type::dispatch(detail::get_array_impl(std::get<I>(m_e), res, args)..., res, args);
        return res;
    }
}

#endif

