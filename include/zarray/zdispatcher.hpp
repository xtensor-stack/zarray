/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZDISPATCHER_HPP
#define XTENSOR_ZDISPATCHER_HPP

#include <xtl/xmultimethods.hpp>

#include "zdispatching_types.hpp"
#include "zmath.hpp"
#include "zreducer_options.hpp"

namespace xt
{
    namespace mpl = xtl::mpl;

    template <class type_list, class undispatched_type_list = mpl::vector<const zassign_args>>
    using zrun_dispatcher_impl = xtl::functor_dispatcher
    <
        type_list,
        void,
        undispatched_type_list,
        xtl::static_caster,
        xtl::basic_fast_dispatcher
    >;

    template <class type_list, class undispatched_type_list = mpl::vector<>>
    using ztype_dispatcher_impl = xtl::functor_dispatcher
    <
        type_list,
        size_t,
        undispatched_type_list,
        xtl::static_caster,
        xtl::basic_fast_dispatcher
    >;

    /**********************
     * zdouble_dispatcher *
     **********************/

    // Double dispatchers are used for unary operations.
    // They dispatch on the single argument and on the
    // result. 
    // Furthermore they are used for the reducers.
    // the dispatch on the single argument of the reducer
    // and the result
    template <class F, class URL = mpl::vector<const zassign_args>, class UTL = mpl::vector<>>
    class zdouble_dispatcher
    {
    public:

        template <class T, class R>
        static void insert();

        template <class T, class R, class... U>
        static void register_dispatching(mpl::vector<mpl::vector<T, R>, U...>);

        static void init();

        // this is a bit of a hack st we can use the same double dispatcher impl
        // independent of the actual undispatched type list
        template<class ... A>
        static void dispatch(const zarray_impl& z1, zarray_impl& res, A && ... );

        // this is a bit of a hack st we can use the same double dispatcher impl
        // independent of the actual undispatched type list
        template<class ... A>
        static size_t get_type_index(const zarray_impl& z1, A && ...);

    private:

        using undispatched_run_type_list = URL;
        using undispatched_type_type_list = UTL;
        static zdouble_dispatcher& instance();

        zdouble_dispatcher();
        ~zdouble_dispatcher() = default;

        template <class T, class R>
        void insert_impl();

        template <class T, class R, class...U>
        inline void register_dispatching_impl(mpl::vector<mpl::vector<T, R>, U...>);
        inline void register_dispatching_impl(mpl::vector<>);

        using zfunctor_type = get_zmapped_functor_t<F>;
        using ztype_dispatcher = ztype_dispatcher_impl<
            mpl::vector<const zarray_impl>,
            undispatched_type_type_list>;
        using zrun_dispatcher = zrun_dispatcher_impl<
            mpl::vector<const zarray_impl, zarray_impl>,
            undispatched_run_type_list>;

        ztype_dispatcher m_type_dispatcher;
        zrun_dispatcher m_run_dispatcher;
    };


    /**********************
     * ztriple_dispatcher *
     **********************/

    // Triple dispatchers are used for binary operations.
    // They dispatch on both arguments and on the result.

    template <class F>
    class ztriple_dispatcher
    {
    public:

        template <class T1, class T2, class R>
        static void insert();

        template <class T1, class T2, class R, class... U>
        static void register_dispatching(mpl::vector<mpl::vector<T1, T2, R>, U...>);

        static void init();
        static void dispatch(const zarray_impl& z1,
                             const zarray_impl& z2,
                             zarray_impl& res,
                             const zassign_args& args);
        static size_t get_type_index(const zarray_impl& z1, const zarray_impl& z2);

    private:
        static ztriple_dispatcher& instance();

        ztriple_dispatcher();
        ~ztriple_dispatcher() = default;

        template <class T1, class T2, class R>
        void insert_impl();

        template <class T1, class T2, class R, class...U>
        inline void register_dispatching_impl(mpl::vector<mpl::vector<T1, T2, R>, U...>);
        inline void register_dispatching_impl(mpl::vector<>);

        using zfunctor_type = get_zmapped_functor_t<F>;
        using ztype_dispatcher = ztype_dispatcher_impl<mpl::vector<const zarray_impl, const zarray_impl>>;
        using zrun_dispatcher = zrun_dispatcher_impl<mpl::vector<const zarray_impl, const zarray_impl, zarray_impl>>;

        ztype_dispatcher m_type_dispatcher;
        zrun_dispatcher m_run_dispatcher;
    };

    /***************
     * zdispatcher *
     ***************/

    template <class F, size_t N, size_t M = 0>
    struct zdispatcher;

    template <class F>
    struct zdispatcher<F, 1>
    {
        using type = zdouble_dispatcher<F>;
    };

    template <class F>
    struct zdispatcher<F, 2>
    {
        using type = ztriple_dispatcher<F>;
    };

    template <class F, size_t N, size_t M = 0>
    using zdispatcher_t = typename zdispatcher<F, N, M>::type;


    template<class F>
    using zreducer_dispatcher = zdouble_dispatcher<F,
        mpl::vector<const zassign_args, const zreducer_options>,
        mpl::vector<const zreducer_options>
    >;

    /************************
     * zarray_impl_register *
     ************************/

    class zarray_impl_register
    {
    public:

        template <class T>
        static void insert();

        static void init();
        static const zarray_impl& get(size_t index);

    private:

        static zarray_impl_register& instance();

        zarray_impl_register();
        ~zarray_impl_register() = default;

        template <class T>
        void insert_impl();

        size_t m_next_index;
        std::vector<std::unique_ptr<zarray_impl>> m_register;
    };

    /****************
     * init_zsystem *
     ****************/

    // Early initialization of all dispatchers
    // and zarray_impl_register
    // return int so it can be assigned to a
    // static variable and be automatically
    // called when loading a shared library
    // for instance.

    int init_zsystem();

    namespace detail
    {
        template <class F>
        struct unary_dispatching_types
        {
            using type = zunary_func_types;
        };

        template <>
        struct unary_dispatching_types<xassign_dummy_functor>
        {
            using type = zunary_ident_types;
        };

        template <>
        struct unary_dispatching_types<xmove_dummy_functor>
        {
            using type = zunary_ident_types;
        };

        template <>
        struct unary_dispatching_types<negate>
        {
            using type = zunary_op_types;
        };

        template <>
        struct unary_dispatching_types<identity>
        {
            using type = zunary_op_types;
        };

        template <>
        struct unary_dispatching_types<bitwise_not>
        {
            using type = mpl::transform_t<build_unary_identity_t, z_int_types>;
        };

        // TODO: replace zunary_classify_types with zunary_bool_func_types
        // when xsimd is fixed
        template <>
        struct unary_dispatching_types<xt::math::isfinite_fun>
        {
            using type = zunary_classify_types;
        };

        template <>
        struct unary_dispatching_types<xt::math::isinf_fun>
        {
            using type = zunary_classify_types;
        };

        template <>
        struct unary_dispatching_types<xt::math::isnan_fun>
        {
            using type = zunary_classify_types;
        };

        template <class F>
        using unary_dispatching_types_t = typename unary_dispatching_types<F>::type;
    }

    /*************************************
     * zdouble_dispatcher implementation *
     *************************************/

    template <class F, class URL, class UTL>
    template <class T, class R>
    inline void zdouble_dispatcher<F,URL, UTL>::insert()
    {
        instance().template insert_impl<T, R>();
    }

    template <class F, class URL, class UTL>
    template <class T, class R, class... U>
    inline void zdouble_dispatcher<F,URL, UTL>::register_dispatching(mpl::vector<mpl::vector<T, R>, U...>)
    {
        instance().register_dispatching_impl(mpl::vector<mpl::vector<T, R>, U...>());
    }

    template <class F, class URL, class UTL>
    inline void zdouble_dispatcher<F,URL, UTL>::init()
    {
        instance();
    }

    // the variance template here is a bit of a hack st. we can use 
    // the same dispatcher impl for the unary operations  and the reducers
    template <class F, class URL, class UTL>
    template<class ... A>
    inline void zdouble_dispatcher<F,URL, UTL>::dispatch(const zarray_impl& z1, zarray_impl& res, A && ... args)
    {
        instance().m_run_dispatcher.dispatch(z1, res, std::forward<A>(args) ...);
    }

    // the variance template here is a bit of a hack st. we can use 
    // the same dispatcher impl for the unary operations  and the reducers
    template <class F, class URL, class UTL>
    template<class ... A>
    inline size_t zdouble_dispatcher<F,URL, UTL>::get_type_index(const zarray_impl& z1,A && ... args)
    {
        return instance().m_type_dispatcher.dispatch(z1, std::forward<A>(args) ...);
    }

    template <class F, class URL, class UTL>
    inline zdouble_dispatcher<F,URL, UTL>& zdouble_dispatcher<F,URL, UTL>::instance()
    {
        static zdouble_dispatcher<F,URL, UTL> inst;
        return inst;
    }

    template <class F, class URL, class UTL>
    inline zdouble_dispatcher<F,URL, UTL>::zdouble_dispatcher()
    {
        register_dispatching_impl(detail::unary_dispatching_types_t<F>());
    }

    template <class F, class URL, class UTL>
    template <class T, class R>
    inline void zdouble_dispatcher<F,URL, UTL>::insert_impl()
    {
        using arg_type = const ztyped_array<T>;
        using res_type = ztyped_array<R>;
        m_run_dispatcher.template insert<arg_type, res_type>(&zfunctor_type::template run<T, R>);
        m_type_dispatcher.template insert<arg_type>(&zfunctor_type::template index<T>);
    }

    template <class F, class URL, class UTL>
    template <class T, class R, class...U>
    inline void zdouble_dispatcher<F,URL, UTL>::register_dispatching_impl(mpl::vector<mpl::vector<T, R>, U...>)
    {
        insert_impl<T, R>();
        register_dispatching_impl(mpl::vector<U...>());
    }

    template <class F, class URL, class UTL>
    inline void zdouble_dispatcher<F,URL, UTL>::register_dispatching_impl(mpl::vector<>)
    {
    }

    /*************************************
     * ztriple_dispatcher implementation *
     *************************************/

    namespace detail
    {
        using zbinary_func_list = mpl::vector
        <
            math::atan2_fun,
            math::hypot_fun,
            math::pow_fun,
            math::fdim_fun,
            math::fmax_fun,
            math::fmin_fun,
            math::remainder_fun,
            math::fmod_fun
        >;

        using zbinary_int_op_list = mpl::vector
        <
            detail::modulus,
            detail::bitwise_and,
            detail::bitwise_or,
            detail::bitwise_xor,
            detail::bitwise_not,
            detail::left_shift,
            detail::right_shift
        >;

        template <class F>
        struct binary_dispatching_types
        {
            using type = std::conditional_t<mpl::contains<zbinary_func_list, F>::value,
                                            zbinary_func_types,
                                            std::conditional_t<mpl::contains<zbinary_int_op_list, F>::value,
                                                               zbinary_int_op_types,
                                                               zbinary_op_types>>;
        };

        template <>
        struct binary_dispatching_types<detail::modulus>
        {
            using type = zbinary_int_op_types;
        };

        template <class F>
        using binary_dispatching_types_t = typename binary_dispatching_types<F>::type;
    }

    template <class F>
    template <class T1, class T2, class R>
    inline void ztriple_dispatcher<F>::insert()
    {
        instance().template insert_impl<T1, T2, R>();
    }

    template <class F>
    template <class T1, class T2, class R, class... U>
    inline void ztriple_dispatcher<F>::register_dispatching(mpl::vector<mpl::vector<T1, T2, R>, U...>)
    {
        instance().register_impl(mpl::vector<mpl::vector<T1, T2, R>, U...>());
    }

    template <class F>
    inline void ztriple_dispatcher<F>::init()
    {
        instance();
    }

    template <class F>
    inline void ztriple_dispatcher<F>::dispatch(const zarray_impl& z1,
                                                const zarray_impl& z2,
                                                zarray_impl& res,
                                                const zassign_args& args)
    {
        instance().m_run_dispatcher.dispatch(z1, z2, res, args);
    }

    template <class F>
    inline size_t ztriple_dispatcher<F>::get_type_index(const zarray_impl& z1, const zarray_impl& z2)
    {
        return instance().m_type_dispatcher.dispatch(z1, z2);
    }

    template <class F>
    inline ztriple_dispatcher<F>& ztriple_dispatcher<F>::instance()
    {
        static ztriple_dispatcher<F> inst;
        return inst;
    }

    template <class F>
    inline ztriple_dispatcher<F>::ztriple_dispatcher()
    {
        register_dispatching_impl(detail::binary_dispatching_types_t<F>());
    }

    template <class F>
    template <class T1, class T2, class R>
    inline void ztriple_dispatcher<F>::insert_impl()
    {
        using arg_type1 = const ztyped_array<T1>;
        using arg_type2 = const ztyped_array<T2>;
        using res_type = ztyped_array<R>;
        m_run_dispatcher.template insert<arg_type1, arg_type2, res_type>(&zfunctor_type::template run<T1, T2, R>);
        m_type_dispatcher.template insert<arg_type1, arg_type1>(&zfunctor_type::template index<T1, T2>);
    }


    template <class F>
    template <class T1, class T2, class R, class...U>
    inline void ztriple_dispatcher<F>::register_dispatching_impl(mpl::vector<mpl::vector<T1, T2, R>, U...>)
    {
        insert_impl<T1, T2, R>();
        register_dispatching_impl(mpl::vector<U...>());
    }

    template <class F>
    inline void ztriple_dispatcher<F>::register_dispatching_impl(mpl::vector<>)
    {
    }

    /***************************************
     * zarray_impl_register implementation *
     ***************************************/

    template <class T>
    inline void zarray_impl_register::insert()
    {
        instance().template insert_impl<T>();
    }

    inline void zarray_impl_register::init()
    {
        instance();
    }

    inline const zarray_impl& zarray_impl_register::get(size_t index)
    {
        return *(instance().m_register[index]);
    }

    inline zarray_impl_register& zarray_impl_register::instance()
    {
        static zarray_impl_register r;
        return r;
    }

    inline zarray_impl_register::zarray_impl_register()
        : m_next_index(0)
    {
        insert_impl<float>();
        insert_impl<double>();
        insert_impl<int32_t>();
        insert_impl<uint32_t>();
        insert_impl<int64_t>();
        insert_impl<uint64_t>();
    }

    template <class T>
    inline void zarray_impl_register::insert_impl()
    {
        size_t& idx = ztyped_array<T>::get_class_static_index();
        if (idx == SIZE_MAX)
        {
            m_register.resize(++m_next_index);
            idx = m_register.size() - 1u;

        }
        else if (m_register.size() <= idx)
        {
            m_register.resize(idx + 1u);
        }
        m_register[idx] = std::unique_ptr<zarray_impl>(detail::build_zarray(std::move(xarray<T>())));
    }

    /******************************
     * init operators dispatchers *
     ******************************/

    template <class T>
    inline int init_zassign_dispatchers()
    {
        zdispatcher_t<detail::xassign_dummy_functor, 1>::init();
        zdispatcher_t<detail::xmove_dummy_functor, 1>::init();
        return 0;
    }

    template <class T>
    inline int init_zarithmetic_dispatchers()
    {
        zdispatcher_t<detail::identity, 1>::init();
        zdispatcher_t<detail::negate, 1>::init();
        zdispatcher_t<detail::plus, 2>::init();
        zdispatcher_t<detail::minus, 2>::init();
        zdispatcher_t<detail::multiplies, 2>::init();
        zdispatcher_t<detail::divides, 2>::init();
        zdispatcher_t<detail::modulus, 2>::init();
        return 0;
    }

    template <class T>
    inline int init_zlogical_dispatchers()
    {
        zdispatcher_t<detail::logical_or, 2>::init();
        zdispatcher_t<detail::logical_and, 2>::init();
        zdispatcher_t<detail::logical_not, 1>::init();
        return 0;
    }

    template <class T>
    inline int init_zbitwise_dispatchers()
    {
        zdispatcher_t<detail::bitwise_or, 2>::init();
        zdispatcher_t<detail::bitwise_and, 2>::init();
        zdispatcher_t<detail::bitwise_xor, 2>::init();
        zdispatcher_t<detail::bitwise_not, 1>::init();
        zdispatcher_t<detail::left_shift, 2>::init();
        zdispatcher_t<detail::right_shift, 2>::init();
        return 0;
    }

    template <class T>
    inline int init_zcomparison_dispatchers()
    {
        zdispatcher_t<detail::less, 2>::init();
        zdispatcher_t<detail::less_equal, 2>::init();
        zdispatcher_t<detail::greater, 2>::init();
        zdispatcher_t<detail::greater_equal, 2>::init();
        zdispatcher_t<detail::equal_to, 2>::init();
        zdispatcher_t<detail::not_equal_to, 2>::init();
        return 0;
    }

    /*************************
     * init math dispatchers *
     *************************/

    template <class T>
    inline int init_zbasic_math_dispatchers()
    {
        zdispatcher_t<math::fabs_fun, 1>::init();
        zdispatcher_t<math::fmod_fun, 2>::init();
        zdispatcher_t<math::remainder_fun, 2>::init();
        zdispatcher_t<math::fmax_fun, 2>::init();
        zdispatcher_t<math::fmin_fun, 2>::init();
        zdispatcher_t<math::fdim_fun, 2>::init();
        return 0;
    }

    template <class T>
    inline int init_zexp_dispatchers()
    {
        zdispatcher_t<math::exp_fun, 1>::init();
        zdispatcher_t<math::exp2_fun, 1>::init();
        zdispatcher_t<math::expm1_fun, 1>::init();
        zdispatcher_t<math::log_fun, 1>::init();
        zdispatcher_t<math::log10_fun, 1>::init();
        zdispatcher_t<math::log2_fun, 1>::init();
        zdispatcher_t<math::log1p_fun, 1>::init();
        return 0;
    }

    template <class T>
    inline int init_zpower_dispatchers()
    {
        zdispatcher_t<math::pow_fun, 2>::init();
        zdispatcher_t<math::sqrt_fun, 1>::init();
        zdispatcher_t<math::cbrt_fun, 1>::init();
        zdispatcher_t<math::hypot_fun, 2>::init();
        return 0;
    }

    template <class T>
    inline int init_ztrigonometric_dispatchers()
    {
        zdispatcher_t<math::sin_fun, 1>::init();
        zdispatcher_t<math::cos_fun, 1>::init();
        zdispatcher_t<math::tan_fun, 1>::init();
        zdispatcher_t<math::asin_fun, 1>::init();
        zdispatcher_t<math::acos_fun, 1>::init();
        zdispatcher_t<math::atan_fun, 1>::init();
        zdispatcher_t<math::atan2_fun, 2>::init();
        return 0;
    }

    template <class T>
    inline int init_zhyperbolic_dispatchers()
    {
        zdispatcher_t<math::sinh_fun, 1>::init();
        zdispatcher_t<math::cosh_fun, 1>::init();
        zdispatcher_t<math::tanh_fun, 1>::init();
        zdispatcher_t<math::asinh_fun, 1>::init();
        zdispatcher_t<math::acosh_fun, 1>::init();
        zdispatcher_t<math::atanh_fun, 1>::init();
        return 0;
    }

    template <class T>
    inline int init_zerf_dispatchers()
    {
        zdispatcher_t<math::erf_fun, 1>::init();
        zdispatcher_t<math::erfc_fun, 1>::init();
        return 0;
    }

    template <class T>
    inline int init_zgamma_dispatchers()
    {
        zdispatcher_t<math::tgamma_fun, 1>::init();
        zdispatcher_t<math::lgamma_fun, 1>::init();
        return 0;
    }

    template <class T>
    inline int init_zrounding_dispatchers()
    {
        zdispatcher_t<math::ceil_fun, 1>::init();
        zdispatcher_t<math::floor_fun, 1>::init();
        zdispatcher_t<math::trunc_fun, 1>::init();
        zdispatcher_t<math::round_fun, 1>::init();
        zdispatcher_t<math::nearbyint_fun, 1>::init();
        zdispatcher_t<math::rint_fun, 1>::init();
        return 0;
    }

    template <class T>
    inline int init_zclassification_dispatchers()
    {
        zdispatcher_t<math::isfinite_fun, 1>::init();
        zdispatcher_t<math::isinf_fun, 1>::init();
        zdispatcher_t<math::isnan_fun, 1>::init();
        return 0;
    }


    template <class T>
    inline int init_zreducer_dispatchers()
    {
        zreducer_dispatcher<zassign_init_value_functor>::init();

        zreducer_dispatcher<zsum_zreducer_functor>::init();
        zreducer_dispatcher<zprod_zreducer_functor>::init();
        zreducer_dispatcher<zmean_zreducer_functor>::init();
        zreducer_dispatcher<zvariance_zreducer_functor>::init();
        zreducer_dispatcher<zstddev_zreducer_functor>::init();
        zreducer_dispatcher<zamin_zreducer_functor>::init();
        zreducer_dispatcher<zamax_zreducer_functor>::init();
        zreducer_dispatcher<znorm_l0_zreducer_functor>::init();
        zreducer_dispatcher<znorm_l1_zreducer_functor>::init();
        return 0;
    }

    /************************************
     * global dispatcher initialization *
     ************************************/

    template <class T>
    inline int init_zoperator_dispatchers()
    {
        init_zassign_dispatchers<T>();
        init_zarithmetic_dispatchers<T>();
        init_zlogical_dispatchers<T>();
        init_zbitwise_dispatchers<T>();
        init_zcomparison_dispatchers<T>();
        return 0;
    }

    template <class T>
    inline int init_zmath_dispatchers()
    {
        init_zbasic_math_dispatchers<T>();
        init_zexp_dispatchers<T>();
        init_zpower_dispatchers<T>();
        init_ztrigonometric_dispatchers<T>();
        init_zhyperbolic_dispatchers<T>();
        init_zerf_dispatchers<T>();
        init_zgamma_dispatchers<T>();
        init_zrounding_dispatchers<T>();
        init_zclassification_dispatchers<T>();
        return 0;
    }

    template <class T = void>
    int init_zsystem()
    {
        init_zreducer_dispatchers<T>();
        init_zoperator_dispatchers<T>();
        init_zmath_dispatchers<T>();
        return 0;
    }
}

#endif
