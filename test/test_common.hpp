#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

#include "test_common_macros.hpp"

#include <utility>
#include <xtensor/xreducer.hpp> // tuple_idx_of


namespace xt
{
    namespace detail
    {
        template< class SEQ, class T>
        struct augment_impl;

        template<std::size_t... I, class T>
        struct augment_impl<std::integer_sequence<std::size_t, I ...>, T>
        {
            template<std::size_t J >
            using helper = std::tuple<std::integral_constant<std::size_t, J>,  std::tuple_element_t<J, T>>;
            using type = std::tuple< helper<I> ...>;
        };
    }

    template<class T>
    struct augment
    {

        using tuple_type = T;
        using tuple_size = std::tuple_size<tuple_type>;
        using iseq = std::make_index_sequence<tuple_size::value>;

        using type = typename detail::augment_impl<iseq, tuple_type>::type;
    };
    template<class T>
    using augment_t = typename augment<T>::type;

    template<class T, class TUPLE>
    inline auto get_param(TUPLE && params)
    {
        using tuple_type = std::decay_t<TUPLE>;
        return std::get<tuple_idx_of<T, augment_t<tuple_type>>::value>(std::forward<TUPLE>(params));
    }

}

#endif

