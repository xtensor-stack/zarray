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

#include <xtl/xoptional.hpp>

#include "zdispatcher.hpp"
#include "zarray_impl_register.hpp"
#include "zarray_buffer_handler.hpp"
#include "zwrappers.hpp"

namespace xt
{
    namespace detail
    {
        using optional_buffer_index_type = xtl::xoptional<std::size_t, bool>;
        using zarray_impl_with_opt_buffer_index =  std::tuple<zarray_impl *, optional_buffer_index_type>;
        using const_zarray_impl_with_opt_buffer_index =  std::tuple<const zarray_impl *, optional_buffer_index_type>;
    }

    template <class F, class... CT>
    class zfunction : public xexpression<zfunction<F, CT...>>
    {
    public:

        using zarray_impl_with_opt_buffer_index = detail::zarray_impl_with_opt_buffer_index;

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
        zarray_impl& assign_to_with_handler(zarray_buffer_handler & res, const zassign_args& args) const;
    private:

        using dispatcher_type = zdispatcher_t<F, sizeof...(CT)>;

        std::size_t compute_dimension() const;

        template <std::size_t... I>
        std::size_t get_result_type_index_impl(std::index_sequence<I...>) const;


       zarray_impl& assign_to_impl(std::index_sequence<0>,   zarray_buffer_handler & buffer_handler, const zassign_args& args) const;
       zarray_impl& assign_to_impl(std::index_sequence<0,1>, zarray_buffer_handler & buffer_handler, const zassign_args& args) const;

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


        // this can be a  zreducer or something similar
        template <class E>
        struct zfunction_argument
        {
            using argument_type = E;

            static std::size_t get_index(const argument_type& e)
            {
                return e.get_result_type_index();
            }

            static auto get_array_impl(const argument_type & e, zarray_buffer_handler & buffer_handler, const zassign_args& args)
            {
                auto buffer_ptr = buffer_handler.get_free_buffer(e.get_result_type_index());
                const auto & array_impl =  e.assign_to(*buffer_ptr, args);
                return std::make_tuple(&array_impl, true);
            }
        };


        template <class F, class... CT>
        struct zfunction_argument<zfunction<F, CT ...>>
        {
            using argument_type = zfunction<F, CT ...>;

            static std::size_t get_index(const argument_type & e)
            {
                return e.get_result_type_index();
            }

            static auto get_array_impl(const argument_type & e,  zarray_buffer_handler & buffer_handler, const zassign_args& args)
            {
                const auto & array_impl = e.assign_to_with_handler(buffer_handler, args);
                return std::make_tuple(&array_impl,true);
            }
        };


        template <>
        struct zfunction_argument<const zarray>
        {
            using argument_type = zarray;
            template <class E>
            static std::size_t get_index(const E& e)
            {
                return e.get_implementation().get_class_index();
            }

            template <class E>
            static auto get_array_impl(const E& e,  zarray_buffer_handler & , const zassign_args&)
            {
                auto & impl = e.get_implementation();
                return std::make_tuple(&impl, false);
            }
        };

        template <>
        struct zfunction_argument<zarray>
        {
            using argument_type = zarray;
            template <class E>
            static std::size_t get_index(const E& e)
            {
                return e.get_implementation().get_class_index();
            }

            template <class E>
            static auto get_array_impl(const E& e,  zarray_buffer_handler & , const zassign_args&)
            {
                auto & impl = e.get_implementation();
                return std::make_tuple(&impl, false);
            }
        };

        template <class CTE>
        struct zfunction_argument<zscalar_wrapper<CTE>>
        {
            using argument_type = zscalar_wrapper<CTE>;

            static std::size_t get_index(const argument_type& e)
            {
                return e.get_class_index();
            }

            static auto get_array_impl(const argument_type& e,  zarray_buffer_handler &, const zassign_args&)
            {
                const zarray_impl & impl = e;
                return std::make_tuple(&impl, false);
            }
        };

        template <class E>
        inline size_t get_result_type_index(const E& e)
        {
            return zfunction_argument<E>::get_index(e);
        }

        template <class E>
        inline auto get_array_impl(const E& e, zarray_buffer_handler & buffer_handler, const zassign_args& args)
        {
            return zfunction_argument<E>::get_array_impl(e, buffer_handler, args);
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
        zarray_buffer_handler  buffer_handler(res);
        auto & r = this->assign_to_with_handler(buffer_handler, args);
        if(&r != &res)
        {
            // this could be refactored in a common function used by zarray itself and here
            zassign_args args;
            args.trivial_broadcast = true;
            if (res.is_chunked())
            {
                // TODO
                throw std::runtime_error("not yet implemented");
            }
            else
            {
                zdispatcher_t<detail::xmove_dummy_functor, 1>::dispatch(res, r, args);
            }
        }
        return res;

    }

    template <class F, class... CT>
    inline zarray_impl& zfunction<F, CT...>::assign_to_with_handler(zarray_buffer_handler & buffer, const zassign_args& args) const
    {
        return assign_to_impl(std::make_index_sequence<sizeof...(CT)>(), buffer, args);
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
    inline zarray_impl& zfunction<F, CT...>::assign_to_impl(std::index_sequence<0>, zarray_buffer_handler & buffer_handler, const zassign_args& args) const
    {

        // the input
        auto impl_and_is_buffer = detail::get_array_impl(std::get<0>(m_e), buffer_handler, args);
        const auto & array_impl = *(std::get<0>(impl_and_is_buffer));
        auto is_buffer = std::get<1>(impl_and_is_buffer);


        // the index type of the output
        const auto result_index_type = this->get_result_type_index();

        // the output
        zarray_impl* result_ptr;

        // only consider buffers
        if(is_buffer && result_index_type == array_impl.get_class_index())
        {
            result_ptr = const_cast<zarray_impl *>(&array_impl);
        }
        else
        {
            // get a FREE buffer
            result_ptr = buffer_handler.get_free_buffer(result_index_type);

            // mark potential input buffer as free
            if(is_buffer)
            {
                buffer_handler.mark_as_free(&array_impl);
            }

        }

        // call the operation dispatcher
        dispatcher_type::dispatch(array_impl, *result_ptr, args);

        return *result_ptr;
    }

    template <class F, class... CT>
    inline zarray_impl& zfunction<F, CT...>::assign_to_impl(std::index_sequence<0,1>, zarray_buffer_handler & buffer_handler, const zassign_args& args) const
    {
        // the inputs
        auto impl_0_and_is_buffer = detail::get_array_impl(std::get<0>(m_e), buffer_handler, args);
        auto impl_1_and_is_buffer = detail::get_array_impl(std::get<1>(m_e), buffer_handler, args);

        const auto & array_impl_0 = *(std::get<0>(impl_0_and_is_buffer));
        const auto & array_impl_1 = *(std::get<0>(impl_1_and_is_buffer));

        auto is_buffer_0 = std::get<1>(impl_0_and_is_buffer);
        auto is_buffer_1 = std::get<1>(impl_1_and_is_buffer);

        // the index type of the output
        const auto result_index_type = this->get_result_type_index();

        // the output
        zarray_impl * result_ptr;


        // only consider buffers
        if(is_buffer_0 && result_index_type == array_impl_0.get_class_index())
        {
            // get the same buffer as non-const!
            result_ptr = const_cast<zarray_impl *>(&array_impl_0);

            // mark potential input buffer as free
            if(is_buffer_1)
            {
                buffer_handler.mark_as_free(&array_impl_1);
            }
        }
        else if(is_buffer_1 && result_index_type == array_impl_1.get_class_index())
        {
            // get the same buffer as non-const!
            result_ptr = const_cast<zarray_impl *>(&array_impl_1);

            // mark potential input buffer as free
            if(is_buffer_0)
            {
                buffer_handler.mark_as_free(&array_impl_0);
            }
        }
        else
        {
            // get a FREE buffer
            result_ptr = buffer_handler.get_free_buffer(result_index_type);

            // mark potential input buffer as free
            if(is_buffer_1)
            {
                buffer_handler.mark_as_free(&array_impl_1);
            }
            if(is_buffer_0)
            {
                buffer_handler.mark_as_free(&array_impl_0);
            }
        }

        // call the operator dispatcher
        dispatcher_type::dispatch(
            array_impl_0,
            array_impl_1,
            *result_ptr, args);

        return *result_ptr;
    }
}

#endif

