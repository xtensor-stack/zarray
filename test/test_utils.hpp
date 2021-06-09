#ifndef XTENSOR_ZTEST_UTILS_HPP
#define XTENSOR_ZTEST_UTILS_HPP

#include "gtest/gtest.h"

#define HETEROGEN_PARAMETRIZED_TEST_SUITE(NAME, TEST_FUNC)\
    template<class T>\
    struct NAME : public TypeHolder<T>\
    {\
    };\
    TYPED_TEST_SUITE(NAME, testing_types_t<augment_t<std::decay_t<decltype(TEST_FUNC())>>>)\

namespace xt
{
    template<typename T>
    struct TypeHolder : public testing::Test
    {
        using type = T;
    };

    template<class T>
    class testing_types_impl;

    template<class ... ARGS>
    class testing_types_impl<std::tuple<ARGS ...>>
    {
    public:
        using type = ::testing::Types<ARGS ...>;
    };

    template<class T>
    using testing_types_t = typename testing_types_impl<T>::type;

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