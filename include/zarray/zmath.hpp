/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZMATH_HPP
#define XTENSOR_ZMATH_HPP

#include "xtensor/xmath.hpp"
#include "xtensor/xnorm.hpp"
#include "zassign.hpp"
#include "zwrappers.hpp"
#include "zreducer_options.hpp"
#include "zmpl.hpp"

namespace xt
{

    template <class XF>
    struct get_zmapped_functor;
    
    template <class XF>
    using get_zmapped_functor_t = typename get_zmapped_functor<XF>::type;

#define XTENSOR_ZMAPPED_FUNCTOR(ZFUN, XFUN)                                        \
    template <>                                                                    \
    struct get_zmapped_functor<XFUN>                                               \
    { using type = ZFUN; }

    namespace detail
    {
        struct xassign_dummy_functor {};
        struct xmove_dummy_functor {};
    }

    struct zassign_functor
    {
        template <class T, class R>
        static void run(const ztyped_array<T>& z, ztyped_array<R>& zres, const zassign_args& args)
        {
            if (!args.chunk_assign)
                zassign_wrapped_expression(zres, z.get_array(), args);
            else
                zassign_wrapped_expression(zres,  z.get_chunk(args.slices()), args);
        }

        template <class T>
        static size_t index(const ztyped_array<T>&)
        {
            return ztyped_array<T>::get_class_static_index();
        }
    };
    XTENSOR_ZMAPPED_FUNCTOR(zassign_functor, detail::xassign_dummy_functor);

    template <class T, class U, class R = void>
    using enable_same_types_t = std::enable_if_t<std::is_same<T, U>::value, R>;

    template <class T, class U, class R = void>
    using disable_same_types_t = std::enable_if_t<!std::is_same<T, U>::value, R>;
    
    struct zmove_functor
    {
        template <class T, class R>
        static disable_same_types_t<T, R> run(const ztyped_array<T>& z, ztyped_array<R>& zres, const zassign_args& args)
        {
            // resize is not called in the move constructor of zarray
            // to avoid useless dyanmic allocation if RHS is about
            // to be moved, therefore we have to call it here.
            zres.resize(z.shape());
            if (!args.chunk_assign)
                zassign_wrapped_expression(zres, z.get_array(), args);
            else
                zassign_wrapped_expression(zres, z.get_chunk(args.slices()), args);
        }

        template <class T, class R>
        static enable_same_types_t<T, R> run(const ztyped_array<T>& z, ztyped_array<R>& zres, const zassign_args& args)
        {
            if (zres.is_array())
            {
                ztyped_array<T>& uz = const_cast<ztyped_array<T>&>(z);
                xarray<T>& ar = uz.get_array();
                zres.get_array() = std::move(ar);
            }
            else if (zres.is_chunked())
            {
                zassign_wrapped_expression(zres, z.get_chunk(args.slices()), args);
            }
            else
            {
                using array_type = ztyped_expression_wrapper<T>;
                array_type& lhs = static_cast<array_type&>(zres);
                ztyped_array<T>& uz = const_cast<ztyped_array<T>&>(z);
                lhs.assign(std::move(uz.get_array()));
            }
        }

        template <class T>
        static size_t index(const ztyped_array<T>&)
        {
            return ztyped_array<T>::get_class_static_index();
        }
    };
    XTENSOR_ZMAPPED_FUNCTOR(zmove_functor, detail::xmove_dummy_functor);

#define XTENSOR_UNARY_ZOPERATOR(ZNAME, XOP, XFUN)                                                  \
    struct ZNAME                                                                                   \
    {                                                                                              \
        template <class T, class R>                                                                \
        static void run(const ztyped_array<T>& z, ztyped_array<R>& zres, const zassign_args& args) \
        {                                                                                          \
            if (!args.chunk_assign)                                                                \
                zassign_wrapped_expression(zres, XOP z.get_array(), args);                         \
            else                                                                                   \
                zassign_wrapped_expression(zres, XOP z.get_chunk(args.slices()), args);            \
        }                                                                                          \
        template <class T>                                                                         \
        static size_t index(const ztyped_array<T>&)                                                \
        {                                                                                          \
            using result_type = ztyped_array<decltype(XOP std::declval<T>())>;                     \
            return result_type::get_class_static_index();                                          \
        }                                                                                          \
    };                                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

#define XTENSOR_BINARY_ZOPERATOR(ZNAME, XOP, XFUN)                                 \
    struct ZNAME                                                                   \
    {                                                                              \
        template <class T1, class T2, class R>                                     \
        static void run(const ztyped_array<T1>& z1,                                \
                        const ztyped_array<T2>& z2,                                \
                        ztyped_array<R>& zres,                                     \
                        const zassign_args& args)                                  \
        {                                                                          \
            if (!args.chunk_assign)                                                \
                zassign_wrapped_expression(zres,                                   \
                                           z1.get_array() XOP z2.get_array(),      \
                                           args);                                  \
            else                                                                   \
                zassign_wrapped_expression(zres,                                   \
                                           z1.get_chunk(args.slices()) XOP z2.get_chunk(args.slices()),      \
                                           args);                                  \
        }                                                                          \
        template <class T1, class T2>                                              \
        static size_t index(const ztyped_array<T1>&, const ztyped_array<T2>&)      \
        {                                                                          \
            using result_type =                                                    \
                ztyped_array<decltype(std::declval<T1>() XOP std::declval<T2>())>; \
            return result_type::get_class_static_index();                          \
        }                                                                          \
    };                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

#define XTENSOR_UNARY_ZFUNCTOR(ZNAME, XEXP, XFUN)                                  \
    struct ZNAME                                                                   \
    {                                                                              \
        template <class T, class R>                                                \
        static void run(const ztyped_array<T>& z,                                  \
                        ztyped_array<R>& zres,                                     \
                        const zassign_args& args)                                  \
        {                                                                          \
            if (!args.chunk_assign )                                               \
                zassign_wrapped_expression(zres, XEXP(z.get_array()), args);       \
            else                                                                   \
                zassign_wrapped_expression(zres, XEXP(z.get_chunk(args.slices())), args); \
        }                                                                          \
        template <class T>                                                         \
        static size_t index(const ztyped_array<T>&)                                \
        {                                                                          \
            using value_type = decltype(std::declval<XFUN>()(std::declval<T>()));  \
            return ztyped_array<value_type>::get_class_static_index();             \
        }                                                                          \
    };                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

#define XTENSOR_BINARY_ZFUNCTOR(ZNAME, XEXP, XFUN)                                 \
    struct ZNAME                                                                   \
    {                                                                              \
        template <class T1, class T2, class R>                                     \
        static void run(const ztyped_array<T1>& z1,                                \
                        const ztyped_array<T2>& z2,                                \
                        ztyped_array<R>& zres,                                     \
                        const zassign_args& args)                                  \
        {                                                                          \
            if (!args.chunk_assign)                                                \
                zassign_wrapped_expression(zres,                                   \
                                           XEXP(z1.get_array(), z2.get_array()),   \
                                           args);                                  \
            else                                                                   \
                zassign_wrapped_expression(zres,                                   \
                                           XEXP(z1.get_chunk(args.slices()), z2.get_chunk(args.slices())),   \
                                           args);                                  \
        }                                                                          \
        template <class T1, class T2>                                              \
        static size_t index(const ztyped_array<T1>&, const ztyped_array<T2>&)      \
        {                                                                          \
            using value_type = decltype(                                           \
                std::declval<XFUN>()(std::declval<T1>(), std::declval<T2>()));     \
            return ztyped_array<value_type>::get_class_static_index();             \
        }                                                                          \
    };                                                                             \
    XTENSOR_ZMAPPED_FUNCTOR(ZNAME, XFUN)

    XTENSOR_UNARY_ZOPERATOR(zidentity, +, detail::identity);
    XTENSOR_UNARY_ZOPERATOR(znegate, -, detail::negate);
    XTENSOR_BINARY_ZOPERATOR(zplus, +, detail::plus);
    XTENSOR_BINARY_ZOPERATOR(zminus, -, detail::minus);
    XTENSOR_BINARY_ZOPERATOR(zmultiuplies, *, detail::multiplies);
    XTENSOR_BINARY_ZOPERATOR(zdivides, /, detail::divides);
    XTENSOR_BINARY_ZOPERATOR(zmodulus, %, detail::modulus);
    XTENSOR_BINARY_ZOPERATOR(zlogical_or, ||, detail::logical_or);
    XTENSOR_BINARY_ZOPERATOR(zlogical_and, &&, detail::logical_and);
    XTENSOR_UNARY_ZOPERATOR(zlogical_not, !, detail::logical_not);
    XTENSOR_BINARY_ZOPERATOR(zbitwise_or, |, detail::bitwise_or);
    XTENSOR_BINARY_ZOPERATOR(zbitwise_and, &, detail::bitwise_and);
    XTENSOR_BINARY_ZOPERATOR(zbitwise_xor, ^, detail::bitwise_xor);
    XTENSOR_UNARY_ZOPERATOR(zbitwise_not, ~, detail::bitwise_not);
    XTENSOR_BINARY_ZOPERATOR(zleft_shift, <<, detail::left_shift);
    XTENSOR_BINARY_ZOPERATOR(zright_shift, >>, detail::right_shift);
    XTENSOR_BINARY_ZOPERATOR(zless, <, detail::less);
    XTENSOR_BINARY_ZOPERATOR(zless_equal, <=, detail::less_equal);
    XTENSOR_BINARY_ZOPERATOR(zgreater, >, detail::greater);
    XTENSOR_BINARY_ZOPERATOR(zgreater_equal, >=, detail::greater_equal);
    XTENSOR_BINARY_ZFUNCTOR(zequal_to, xt::equal, detail::equal_to);
    XTENSOR_BINARY_ZFUNCTOR(znot_equal_to, xt::not_equal, detail::not_equal_to);


    XTENSOR_UNARY_ZFUNCTOR(zfabs, xt::fabs, math::fabs_fun);
    XTENSOR_BINARY_ZFUNCTOR(zfmod, xt::fmod, math::fmod_fun);
    XTENSOR_BINARY_ZFUNCTOR(zremainder, xt::remainder, math::remainder_fun);
    //XTENSOR_TERNARY_ZFUNCTOR(fma);
    XTENSOR_BINARY_ZFUNCTOR(zfmax, xt::fmax, math::fmax_fun);
    XTENSOR_BINARY_ZFUNCTOR(zfmin, xt::fmin, math::fmin_fun);
    XTENSOR_BINARY_ZFUNCTOR(zfdim, xt::fdim, math::fdim_fun);
    XTENSOR_UNARY_ZFUNCTOR(zexp, xt::exp, math::exp_fun);
    XTENSOR_UNARY_ZFUNCTOR(zexp2, xt::exp2, math::exp2_fun);
    XTENSOR_UNARY_ZFUNCTOR(zexpm1, xt::expm1, math::expm1_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog, xt::log, math::log_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog10, xt::log10, math::log10_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog2, xt::log2, math::log2_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlog1p, xt::log1p, math::log1p_fun);
    XTENSOR_BINARY_ZFUNCTOR(zpow, xt::pow, math::pow_fun);
    XTENSOR_UNARY_ZFUNCTOR(zsqrt, xt::sqrt, math::sqrt_fun);
    XTENSOR_UNARY_ZFUNCTOR(zcbrt, xt::cbrt, math::cbrt_fun);
    XTENSOR_BINARY_ZFUNCTOR(zhypot, xt::hypot, math::hypot_fun);
    XTENSOR_UNARY_ZFUNCTOR(zsin, xt::sin, math::sin_fun);
    XTENSOR_UNARY_ZFUNCTOR(zcos, xt::cos, math::cos_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztan, xt::tan, math::tan_fun);
    XTENSOR_UNARY_ZFUNCTOR(zasin, xt::asin, math::asin_fun);
    XTENSOR_UNARY_ZFUNCTOR(zacos, xt::acos, math::acos_fun);
    XTENSOR_UNARY_ZFUNCTOR(zatan, xt::atan, math::atan_fun);
    XTENSOR_BINARY_ZFUNCTOR(zatan2, xt::atan2, math::atan2_fun);
    XTENSOR_UNARY_ZFUNCTOR(zsinh, xt::sinh, math::sinh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zcosh, xt::cosh, math::cosh_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztanh, xt::tanh, math::tanh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zasinh, xt::asinh, math::asinh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zacosh, xt::acosh, math::acosh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zatanh, xt::atanh, math::atanh_fun);
    XTENSOR_UNARY_ZFUNCTOR(zerf, xt::erf, math::erf_fun);
    XTENSOR_UNARY_ZFUNCTOR(zerfc, xt::erfc, math::erfc_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztgamma, xt::tgamma, math::tgamma_fun);
    XTENSOR_UNARY_ZFUNCTOR(zlgamma, xt::lgamma, math::lgamma_fun);
    XTENSOR_UNARY_ZFUNCTOR(zceil, xt::ceil, math::ceil_fun);
    XTENSOR_UNARY_ZFUNCTOR(zfloor, xt::floor, math::floor_fun);
    XTENSOR_UNARY_ZFUNCTOR(ztrunc, xt::trunc, math::trunc_fun);
    XTENSOR_UNARY_ZFUNCTOR(zround, xt::round, math::round_fun);
    XTENSOR_UNARY_ZFUNCTOR(znearbyint, xt::nearbyint, math::nearbyint_fun);
    XTENSOR_UNARY_ZFUNCTOR(zrint, xt::rint, math::rint_fun);
    XTENSOR_UNARY_ZFUNCTOR(zisfinite, xt::isfinite, math::isfinite_fun);
    XTENSOR_UNARY_ZFUNCTOR(zisinf, xt::isinf, math::isinf_fun);
    XTENSOR_UNARY_ZFUNCTOR(zisnan, xt::isnan, math::isnan_fun);


    // forward declaration
    template<class F, class E, typename detail::enable_zarray_t<std::decay_t<E>> * = nullptr>
    auto make_zreducer(E && e, const zreducer_options & options);

    // forward declaration
    class zreducer_options;

    struct zassign_init_value_functor
    {

        template <class T, class R>
        static void run(
            const ztyped_array<T>& z,
            ztyped_array<R>& zres,
            const zassign_args& args,
            const zreducer_options&
        );
        template <class T>
        static std::size_t index(const ztyped_array<T>&, const zreducer_options& );
    };
    XTENSOR_ZMAPPED_FUNCTOR(zassign_init_value_functor, zassign_init_value_functor);

    template <class T, class R>
    inline void zassign_init_value_functor::run
    (
        const ztyped_array<T>& z,
        ztyped_array<R>& zres,
        const zassign_args& args,
        const zreducer_options&
    )
    {
        if (!args.chunk_assign)
        {
            // write the initial value which is wrapped in a zscalar_wrapper 
            // to the first position of the result array
            *(zres.get_array().begin()) = static_cast<R>(*(z.get_array().begin()));
        }
        else
        {
            auto init_value = static_cast<R>(*(z.get_array().begin()));

            auto & chunked_array = dynamic_cast<ztyped_chunked_array<R>&>(zres);
            auto chunk_iter = args.chunk_iter;
            auto shape = chunked_array.chunk_shape();
            auto tmp = xarray<R>::from_shape(shape);
            tmp.fill(init_value);
            chunked_array.assign_chunk(std::move(tmp), chunk_iter);
        }
    }
    template <class T>
    inline std::size_t zassign_init_value_functor::index(const ztyped_array<T>&, const zreducer_options& )
    {
        throw std::runtime_error("should be unreachable");
        return 0;
    }

    template<class F>
    struct zreducer_functor
    {
        template <class T, class R>
        static void run (const ztyped_array<T>& in,ztyped_array<R>& zres,
            const zassign_args& assign_args,const zreducer_options& options
        );
        template <class T>
        static size_t index(const ztyped_array<T>& in, const zreducer_options& options );
    };

    template<class F>
    template <class T, class R>
    inline void zreducer_functor<F>::run
    (
        const ztyped_array<T>& input_array,
        ztyped_array<R>& zres,
        const zassign_args& assign_args,
        const zreducer_options& options
    )
    {
        if (!assign_args.chunk_assign)
        {
            options.visit_reducer_options<T>(false /*force_lazy*/,[&assign_args, &input_array, &zres](auto&&... reduce_args)
            {
                auto res_expr = F::run(input_array.get_array(), std::forward<decltype(reduce_args)>(reduce_args)...);
                zassign_wrapped_expression(zres, std::move(res_expr), assign_args);
            });
        }
        else
        {
            options.visit_reducer_options<T>(true /*force_lazy*/, [&assign_args, &input_array, &zres](auto&&... reduce_args)
            {
                auto res_expr = F::run(input_array.get_array(), std::forward<decltype(reduce_args)>(reduce_args)...);
                auto chunk_res = xt::strided_view(std::move(res_expr), assign_args.slices());
                zassign_wrapped_expression(zres, std::move(chunk_res), assign_args);
            });
        }
    }

    template<class F>
    template <class T>
    inline std::size_t zreducer_functor<F>::index
    (
        const ztyped_array<T>& input_array,
        const zreducer_options& options
    )
    {
        std::size_t result;
        options.visit_reducer_options([&result, &input_array](auto&&... reduce_args)
        {
            using expr_type = decltype(F::run(input_array.get_array(), std::forward<decltype(reduce_args)>(reduce_args)...));
            using expr_value_type = typename std::decay_t<expr_type>::value_type;
            result = ztyped_array<expr_value_type>::get_class_static_index();
        });
        return result;
    }

    #define  XTENSOR_ZREDUCER_FUNCTOR_HELPER(FUNCTOR_NAME, FUNC_NAME)\
    namespace detail\
    {\
        struct FUNCTOR_NAME ## _helper\
        {\
            template<class ... T>\
            auto static run(T && ... args) { return FUNC_NAME(std::forward<T>(args) ...);}\
        };\
    } \
    struct FUNCTOR_NAME : zreducer_functor<detail:: FUNCTOR_NAME ## _helper>{ \
    };\
    XTENSOR_ZMAPPED_FUNCTOR(FUNCTOR_NAME, FUNCTOR_NAME);\
    namespace zt\
    {\
        template<class E, class A, class EVS = DEFAULT_STRATEGY_REDUCERS,\
             XTL_REQUIRES(detail::has_zexpression_tag<E>)\
        >\
        inline auto FUNC_NAME(E && e, std::initializer_list<A> axis, EVS && options = EVS())\
        {\
            zreducer_options zoptions(axis, std::forward<EVS>(options));\
            return make_zreducer<FUNCTOR_NAME>(std::forward<E>(e), zoptions);\
        }\
        template<class E, class A, class EVS = DEFAULT_STRATEGY_REDUCERS,\
             XTL_REQUIRES(xtl::negation<is_reducer_options<A>>, detail::has_zexpression_tag<E>)\
        >\
        inline auto FUNC_NAME(E && e, A && axis, EVS && options = EVS())\
        {\
            zreducer_options zoptions(std::forward<A>(axis), std::forward<EVS>(options));\
            return make_zreducer<FUNCTOR_NAME>(std::forward<E>(e), zoptions);\
        }\
        template<class E, class EVS = DEFAULT_STRATEGY_REDUCERS,\
             XTL_REQUIRES(is_reducer_options<EVS>, detail::has_zexpression_tag<E>)\
        >\
        inline auto FUNC_NAME(E && e, EVS && options = EVS())\
        {\
            auto axis = dynamic_shape<std::size_t>(e.dimension());\
            std::iota(axis.begin(), axis.end(), 0);\
            zreducer_options zoptions(std::move(axis), options);\
            return make_zreducer<FUNCTOR_NAME>(std::forward<E>(e), zoptions);\
        }\
    }

    XTENSOR_ZREDUCER_FUNCTOR_HELPER(zsum_zreducer_functor,      sum)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(zprod_zreducer_functor,     prod)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(zmean_zreducer_functor,     mean)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(zvariance_zreducer_functor, variance)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(zstddev_zreducer_functor,   stddev)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(zamax_zreducer_functor,     amax)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(zamin_zreducer_functor,     amin)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(znorm_l0_zreducer_functor, norm_l0)
    XTENSOR_ZREDUCER_FUNCTOR_HELPER(znorm_l1_zreducer_functor, norm_l1)



#undef XTENSOR_BINARY_ZFUNCTOR
#undef XTENSOR_UNARY_ZFUNCTOR
#undef XTENSOR_BINARY_ZOPERATOR
#undef XTENSOR_UNARY_ZOPERATOR
#undef XTENSOR_ZMAPPED_FUNCTOR
#undef XTENSOR_ZREDUCER_FUNCTOR_HELPER





}

#endif
