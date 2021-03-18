/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZARRAY_IMPL_HPP
#define XTENSOR_ZARRAY_IMPL_HPP

#include <xtl/xplatform.hpp>
#include <xtl/xhalf_float.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xchunked_array.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xscalar.hpp>
#include <xtensor/xshape.hpp>
#include <xtensor/xshape.hpp>
#include "xtensor/xstrided_view.hpp"

namespace xt
{
    template <class CTE>
    class zarray_wrapper;

    template <class CTE>
    class zchunked_wrapper;

    template <class CTE>
    class zexpression_wrapper;

    namespace detail
    {
        template <class E>
        struct is_xstrided_view : std::false_type
        {
        };

        template <class CT, class S, layout_type L, class FST>
        struct is_xstrided_view<xstrided_view<CT, S, L, FST>> : std::true_type
        {
        };

        template <class CT>
        using is_const = std::is_const<std::remove_reference_t<CT>>;

        template <class CT, class R = void>
        using disable_xstrided_view_t = std::enable_if_t<!is_xstrided_view<CT>::value || is_const<CT>::value, R>;

        template <class CT, class R = void>
        using enable_const_t = std::enable_if_t<is_const<CT>::value, R>;

        template <class CT, class R = void>
        using disable_const_t = std::enable_if_t<!is_const<CT>::value, R>;
    }

    /******************
     * zarray builder *
     ******************/

    namespace detail
    {
        template <class E>
        struct is_xarray : std::false_type
        {
        };

        template <class T, layout_type L, class A, class SA>
        struct is_xarray<xarray<T, L, A, SA>> : std::true_type
        {
        };

        template <class E>
        struct is_chunked_array : std::false_type
        {
        };

        template <class CS>
        struct is_chunked_array<xchunked_array<CS>> : std::true_type
        {
        };

        template <class E>
        struct zwrapper_builder
        {
            using closure_type = xtl::closure_type_t<E>;
            using wrapper_type = std::conditional_t<is_xarray<std::decay_t<E>>::value,
                                                    zarray_wrapper<closure_type>,
                                                    std::conditional_t<is_chunked_array<std::decay_t<E>>::value,
                                                                       zchunked_wrapper<closure_type>,
                                                                       zexpression_wrapper<closure_type>
                                                                      >
                                                    >;

            template <class OE>
            static wrapper_type* run(OE&& e)
            {
                return new wrapper_type(std::forward<OE>(e));
            }
        };

        template <class E>
        inline auto build_zarray(E&& e)
        {
            return zwrapper_builder<E>::run(std::forward<E>(e));
        }
    }

    const std::string endianness_string = (xtl::endianness() == xtl::endian::little_endian) ? "<" : ">";

    template <class T>
    inline void set_data_type(nlohmann::json& metadata)
    {
    }

    template <>
    inline void set_data_type<bool>(nlohmann::json& metadata)
    {
        metadata["data_type"] = "bool";
    }

    template <>
    inline void set_data_type<uint8_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = "u1";
    }

    template <>
    inline void set_data_type<int8_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = "i1";
    }

    template <>
    inline void set_data_type<int16_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "i2";
    }

    template <>
    inline void set_data_type<uint16_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "u2";
    }

    template <>
    inline void set_data_type<int32_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "i4";
    }

    template <>
    inline void set_data_type<uint32_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "u4";
    }

    template <>
    inline void set_data_type<int64_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "i8";
    }

    template <>
    inline void set_data_type<uint64_t>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "u8";
    }

    template <>
    inline void set_data_type<xtl::half_float>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "f2";
    }

    template <>
    inline void set_data_type<float>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "f4";
    }

    template <>
    inline void set_data_type<double>(nlohmann::json& metadata)
    {
        metadata["data_type"] = endianness_string + "f8";
    }

    /*************************
     * zarray_expression_tag *
     *************************/

    struct zarray_expression_tag {};

    namespace extension
    {
        template <>
        struct expression_tag_and<xtensor_expression_tag, zarray_expression_tag>
        {
            using type = zarray_expression_tag;
        };

        template <>
        struct expression_tag_and<zarray_expression_tag, xtensor_expression_tag>
            : expression_tag_and<xtensor_expression_tag, zarray_expression_tag>
        {
        };

        template <>
        struct expression_tag_and<zarray_expression_tag, zarray_expression_tag>
        {
            using type = zarray_expression_tag;
        };
    }

    /***************
     * zarray_impl *
     ***************/

    class zarray_impl
    {
    public:

        using self_type = zarray_impl;
        using shape_type = dynamic_shape<std::size_t>;

        virtual ~zarray_impl() = default;

        zarray_impl(zarray_impl&&) = delete;
        zarray_impl& operator=(const zarray_impl&) = delete;
        zarray_impl& operator=(zarray_impl&&) = delete;

        virtual self_type* clone() const = 0;

        virtual self_type* strided_view(xstrided_slice_vector& slices) = 0;

        virtual const nlohmann::json& get_metadata() const = 0;
        virtual void set_metadata(const nlohmann::json& metadata) = 0;
        virtual std::size_t dimension() const = 0;
        virtual const shape_type& shape() const = 0;
        virtual void resize(const shape_type& shape) = 0;
        virtual void resize(shape_type&& shape) = 0;
        virtual bool broadcast_shape(shape_type& shape, bool reuse_cache = 0) const = 0;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    protected:

        zarray_impl() = default;
        zarray_impl(const zarray_impl&) = default;
    };

    /****************
     * ztyped_array *
     ****************/

    template <class T>
    class ztyped_array : public zarray_impl
    {
    public:

        using base_type = zarray_impl;
        using shape_type = base_type::shape_type;

        virtual ~ztyped_array() = default;

        virtual bool is_array() const = 0;

        virtual xarray<T>& get_array() = 0;
        virtual const xarray<T>& get_array() const = 0;

        virtual void assign(xarray<T>&& rhs) = 0;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    protected:

        ztyped_array() = default;
        ztyped_array(const ztyped_array&) = default;
    };

    /***********************
     * zexpression_wrapper *
     ***********************/

    template <class CTE>
    class zexpression_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>
    {
    public:

        using self_type = zexpression_wrapper<CTE>;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;
        using shape_type = typename base_type::shape_type;

        template <class E>
        zexpression_wrapper(E&& e);

        virtual ~zexpression_wrapper() = default;

        bool is_array() const override;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        void assign(xarray<value_type>&& rhs) override;

        self_type* clone() const override;

        zarray_impl* strided_view(xstrided_slice_vector& slices) override;

        const nlohmann::json& get_metadata() const override;
        void set_metadata(const nlohmann::json& metadata) override;
        std::size_t dimension() const override;
        const shape_type& shape() const override;
        void resize(const shape_type&) override;
        void resize(shape_type&&) override;
        bool broadcast_shape(shape_type& shape, bool reuse_cache = 0) const override;

    private:

        zexpression_wrapper(const zexpression_wrapper&) = default;

        void compute_cache() const;

        zarray_impl* strided_view_impl(xstrided_slice_vector& slices, std::true_type);
        zarray_impl* strided_view_impl(xstrided_slice_vector& slices, std::false_type);

        template <class CT>
        using enable_assignable_t = enable_assignable_expression<CT, xarray<value_type>>;

        template <class CT>
        using enable_not_assignable_t = enable_not_assignable_expression<CT, xarray<value_type>>;

        template <class CT = CTE>
        enable_assignable_t<CT> assign_impl(xarray<value_type>&& rhs);
        
        template <class CT = CTE>   
        enable_not_assignable_t<CT> assign_impl(xarray<value_type>&& rhs);

        template <class CT = CTE>
        enable_assignable_t<CT> resize_impl();

        template <class CT = CTE>
        enable_not_assignable_t<CT> resize_impl();

        CTE m_expression;
        mutable xarray<value_type> m_cache;
        mutable bool m_cache_initialized;
        nlohmann::json m_metadata;
    };

    /*******************
     * zscalar_wrapper *
     *******************/

    template <class CTE>
    class zscalar_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>
    {
    public:

        using self_type = zscalar_wrapper;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;
        using shape_type = typename base_type::shape_type;

        template <class E>
        zscalar_wrapper(E&& e);

        zscalar_wrapper(zscalar_wrapper&&) = default;

        virtual ~zscalar_wrapper() = default;

        bool is_array() const override;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        void assign(xarray<value_type>&& rhs) override;

        self_type* clone() const override;

        zarray_impl* strided_view(xstrided_slice_vector& slices) override;

        const nlohmann::json& get_metadata() const override;
        void set_metadata(const nlohmann::json& metadata) override;
        std::size_t dimension() const override;
        const shape_type& shape() const override;
        void resize(const shape_type&) override;
        void resize(shape_type&&) override;
        bool broadcast_shape(shape_type& shape, bool reuse_cache = 0) const override;

    private:

        zscalar_wrapper(const zscalar_wrapper&) = default;
        
        CTE m_expression;
        xarray<value_type> m_array;
        nlohmann::json m_metadata;
    };

    /******************
     * zarray_wrapper *
     ******************/

    template <class CTE>
    class zarray_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>
    {
    public:

        using self_type = zarray_wrapper;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;
        using shape_type = typename base_type::shape_type;

        template <class E>
        zarray_wrapper(E&& e);

        virtual ~zarray_wrapper() = default;

        bool is_array() const override;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        void assign(xarray<value_type>&& rhs) override;

        self_type* clone() const override;

        zarray_impl* strided_view(xstrided_slice_vector& slices) override;

        const nlohmann::json& get_metadata() const override;
        void set_metadata(const nlohmann::json& metadata) override;
        std::size_t dimension() const override;
        const shape_type& shape() const override;
        void resize(const shape_type&) override;
        void resize(shape_type&&) override;
        bool broadcast_shape(shape_type& shape, bool reuse_cache = 0) const override;

    private:

        zarray_wrapper(const zarray_wrapper&) = default;

        template <class CT = CTE>
        detail::enable_const_t<CT> assign_impl(xarray<value_type>&& rhs);

        template <class CT = CTE>
        detail::disable_const_t<CT> assign_impl(xarray<value_type>&& rhs);

        CTE m_array;
        nlohmann::json m_metadata;
    };

    /********************
     * zchunked_wrapper *
     ********************/

    class zchunked_array
    {
    public:

        using shape_type = xt::dynamic_shape<std::size_t>;

        virtual ~zchunked_array() = default;
        virtual const shape_type& chunk_shape() const = 0;
    };

    template <class CTE>
    class zchunked_wrapper : public ztyped_array<typename std::decay_t<CTE>::value_type>,
                             public zchunked_array
    {
    public:

        using self_type = zchunked_wrapper;
        using value_type = typename std::decay_t<CTE>::value_type;
        using base_type = ztyped_array<value_type>;
        using shape_type = zchunked_array::shape_type;

        template <class E>
        zchunked_wrapper(E&& e);

        virtual ~zchunked_wrapper() = default;

        bool is_array() const override;

        xarray<value_type>& get_array() override;
        const xarray<value_type>& get_array() const override;

        void assign(xarray<value_type>&& rhs) override;

        self_type* clone() const override;

        zarray_impl* strided_view(xstrided_slice_vector& slices) override;

        const nlohmann::json& get_metadata() const override;
        void set_metadata(const nlohmann::json& metadata) override;
        std::size_t dimension() const override;
        const shape_type& shape() const override;
        void resize(const shape_type& shape) override;
        void resize(shape_type&&) override;
        bool broadcast_shape(shape_type& shape, bool reuse_cache = 0) const override;

        const shape_type& chunk_shape() const override;

    private:

        zchunked_wrapper(const zchunked_wrapper&) = default;

        void compute_cache() const;

        template <class CT = CTE>
        detail::enable_const_t<CT> assign_impl(xarray<value_type>&& rhs);

        template <class CT = CTE>
        detail::disable_const_t<CT> assign_impl(xarray<value_type>&& rhs);
        
        CTE m_chunked_array;
        shape_type m_chunk_shape;
        mutable xarray<value_type> m_cache;
        mutable bool m_cache_initialized;
        mutable dynamic_shape<std::ptrdiff_t> m_strides;
        mutable bool m_strides_initialized;

        nlohmann::json m_metadata;

    };

    /***********************
     * zexpression_wrapper *
     ***********************/

    template <class CTE>
    template <class E>
    inline zexpression_wrapper<CTE>::zexpression_wrapper(E&& e)
        : base_type()
        , m_expression(std::forward<E>(e))
        , m_cache()
        , m_cache_initialized(false)
    {
        set_data_type<value_type>(m_metadata);
    }

    template <class CTE>
    bool zexpression_wrapper<CTE>::is_array() const
    {
        return false;
    }

    template <class CTE>
    auto zexpression_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    auto zexpression_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    void zexpression_wrapper<CTE>::assign(xarray<value_type>&& rhs)
    {
        assign_impl(std::move(rhs));
    }
    
    template <class CTE>
    auto zexpression_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    template <class CTE>
    zarray_impl* zexpression_wrapper<CTE>::strided_view(xstrided_slice_vector& slices)
    {
        return strided_view_impl(slices, detail::is_xstrided_view<CTE>());
    }

    template <class CTE>
    inline zarray_impl* zexpression_wrapper<CTE>::strided_view_impl(xstrided_slice_vector& slices, std::true_type)
    {
        auto e = xt::strided_view(get_array(), slices);
        return detail::build_zarray(std::move(e));
    }

    template <class CTE>
    inline zarray_impl* zexpression_wrapper<CTE>::strided_view_impl(xstrided_slice_vector& slices, std::false_type)
    {
        auto e = xt::strided_view(m_expression, slices);
        return detail::build_zarray(std::move(e));
    }

    template <class CTE>
    auto zexpression_wrapper<CTE>::get_metadata() const -> const nlohmann::json&
    {
        return m_metadata;
    }

    template <class CTE>
    void zexpression_wrapper<CTE>::set_metadata(const nlohmann::json& metadata)
    {
        m_metadata = metadata;
    }

    template <class CTE>
    std::size_t zexpression_wrapper<CTE>::dimension() const
    {
        return m_expression.dimension();
    }

    template <class CTE>
    auto zexpression_wrapper<CTE>::shape() const -> const shape_type&
    {
        compute_cache();
        return m_cache.shape();
    }

    template <class CTE>
    void zexpression_wrapper<CTE>::resize(const shape_type&)
    {
        resize_impl();
    }

    template <class CTE>
    void zexpression_wrapper<CTE>::resize(shape_type&&)
    {
        resize_impl();
    }

    template <class CTE>
    bool zexpression_wrapper<CTE>::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        return m_expression.broadcast_shape(shape, reuse_cache);
    }

    template <class CTE>
    inline void zexpression_wrapper<CTE>::compute_cache() const
    {
        if (!m_cache_initialized)
        {
            noalias(m_cache) = m_expression;
            m_cache_initialized = true;
        }
    }
    
    template <class CTE>
    template <class CT>
    inline auto zexpression_wrapper<CTE>::assign_impl(xarray<value_type>&& rhs) -> enable_assignable_t<CT>
    {
        m_expression = rhs;
    }
    
    template <class CTE>
    template <class CT>   
    inline auto zexpression_wrapper<CTE>::assign_impl(xarray<value_type>&&) -> enable_not_assignable_t<CT>
    {
        throw std::runtime_error("unevaluated expression is not assignable");
    }
    
    template <class CTE>
    template <class CT>
    inline auto zexpression_wrapper<CTE>::resize_impl() -> enable_assignable_t<CT>
    {
        // Only wrappers on views are assignable. Resizing is a no op.
    }

    template <class CTE>
    template <class CT>
    inline auto zexpression_wrapper<CTE>::resize_impl() -> enable_not_assignable_t<CT>
    {
        throw std::runtime_error("cannot resize not assignable expression wrapper");
    }
    
    /*******************
     * zscalar_wrapper *
     *******************/

    template <class CTE>
    template <class E>
    inline zscalar_wrapper<CTE>::zscalar_wrapper(E&& e)
        : base_type()
        , m_expression(std::forward<E>(e))
        , m_array(m_expression())
    {
        set_data_type<value_type>(m_metadata);
    }

    template <class CTE>
    bool zscalar_wrapper<CTE>::is_array() const
    {
        return false;
    }

    template <class CTE>
    auto zscalar_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        return m_array;
    }

    template <class CTE>
    auto zscalar_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        return m_array;
    }

    template <class CTE>
    void zscalar_wrapper<CTE>::assign(xarray<value_type>&&)
    {
        throw std::runtime_error("scalar cannot be assigned an array");
    }

    template <class CTE>
    auto zscalar_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    template <class CTE>
    zarray_impl* zscalar_wrapper<CTE>::strided_view(xstrided_slice_vector& slices)
    {
        auto e = xt::strided_view(m_array, slices);
        return detail::build_zarray(std::move(e));
    }

    template <class CTE>
    auto zscalar_wrapper<CTE>::get_metadata() const -> const nlohmann::json&
    {
        return m_metadata;
    }

    template <class CTE>
    void zscalar_wrapper<CTE>::set_metadata(const nlohmann::json& metadata)
    {
        m_metadata = metadata;
    }

    template <class CTE>
    std::size_t zscalar_wrapper<CTE>::dimension() const
    {
        return m_array.dimension();
    }

    template <class CTE>
    auto zscalar_wrapper<CTE>::shape() const -> const shape_type&
    {
        return m_array.shape();
    }

    template <class CTE>
    void zscalar_wrapper<CTE>::resize(const shape_type&)
    {
        throw std::runtime_error("Cannot resize scalar wrapper");
    }

    template <class CTE>
    void zscalar_wrapper<CTE>::resize(shape_type&&)
    {
        throw std::runtime_error("Cannot resize scalar wrapper");
    }

    template <class CTE>
    bool zscalar_wrapper<CTE>::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        return m_array.broadcast_shape(shape, reuse_cache);
    }
    
    /******************
     * zarray_wrapper *
     ******************/

    namespace detail
    {
        template <class T>
        struct zarray_wrapper_helper
        {
            static inline xarray<T>& get_array(xarray<T>& ar)
            {
                return ar;
            }

            static inline xarray<T>& get_array(const xarray<T>&)
            {
                throw std::runtime_error("Cannot return non const array from const array");
            }

            template <class S>
            static inline void resize(xarray<T>& ar, S&& shape)
            {
                ar.resize(std::forward<S>(shape));
            }

            template <class S>
            static inline void resize(const xarray<T>&, S&&)
            {
                throw std::runtime_error("Cannot resize const array");
            }
        };
    }

    template <class CTE>
    template <class E>
    inline zarray_wrapper<CTE>::zarray_wrapper(E&& e)
        : base_type()
        , m_array(std::forward<E>(e))
    {
        set_data_type<value_type>(m_metadata);
    }

    template <class CTE>
    bool zarray_wrapper<CTE>::is_array() const
    {
        return true;
    }

    template <class CTE>
    auto zarray_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        return detail::zarray_wrapper_helper<value_type>::get_array(m_array);
    }

    template <class CTE>
    auto zarray_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        return m_array;
    }

    template <class CTE>
    void zarray_wrapper<CTE>::assign(xarray<value_type>&& rhs)
    {
        assign_impl(std::move(rhs));
    }

    template <class CTE>
    auto zarray_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    template <class CTE>
    zarray_impl* zarray_wrapper<CTE>::strided_view(xstrided_slice_vector& slices)
    {
        auto e = xt::strided_view(m_array, slices);
        return detail::build_zarray(std::move(e));
    }

    template <class CTE>
    auto zarray_wrapper<CTE>::get_metadata() const -> const nlohmann::json&
    {
        return m_metadata;
    }

    template <class CTE>
    void zarray_wrapper<CTE>::set_metadata(const nlohmann::json& metadata)
    {
        m_metadata = metadata;
    }

    template <class CTE>
    std::size_t zarray_wrapper<CTE>::dimension() const
    {
        return m_array.dimension();
    }

    template <class CTE>
    auto zarray_wrapper<CTE>::shape() const -> const shape_type&
    {
        return m_array.shape();
    }

    template <class CTE>
    void zarray_wrapper<CTE>::resize(const shape_type& shape)
    {
        detail::zarray_wrapper_helper<value_type>::resize(m_array, shape);
    }

    template <class CTE>
    void zarray_wrapper<CTE>::resize(shape_type&& shape)
    {
        detail::zarray_wrapper_helper<value_type>::resize(m_array, std::move(shape));
    }

    template <class CTE>
    bool zarray_wrapper<CTE>::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        return m_array.broadcast_shape(shape, reuse_cache);
    }
    
    template <class CTE>
    template <class CT>
    inline detail::enable_const_t<CT> zarray_wrapper<CTE>::assign_impl(xarray<value_type>&&)
    {
        throw std::runtime_error("const array is not assignable");
    }

    template <class CTE>
    template <class CT>
    inline detail::disable_const_t<CT> zarray_wrapper<CTE>::assign_impl(xarray<value_type>&& rhs)
    {
        m_array = std::move(rhs);
    }
   
    /********************
     * zchunked_wrapper *
     ********************/

    template <class CTE>
    template <class E>
    inline zchunked_wrapper<CTE>::zchunked_wrapper(E&& e)
        : base_type()
        , m_chunked_array(std::forward<E>(e))
        , m_chunk_shape(m_chunked_array.chunk_shape().size())
        , m_cache()
        , m_cache_initialized(false)
        , m_strides_initialized(false)
    {
        std::copy(m_chunked_array.chunk_shape().begin(),
                  m_chunked_array.chunk_shape().end(),
                  m_chunk_shape.begin());
        set_data_type<value_type>(m_metadata);
    }

    template <class CTE>
    bool zchunked_wrapper<CTE>::is_array() const
    {
        return false;
    }

    template <class CTE>
    auto zchunked_wrapper<CTE>::get_array() -> xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    auto zchunked_wrapper<CTE>::get_array() const -> const xarray<value_type>&
    {
        compute_cache();
        return m_cache;
    }

    template <class CTE>
    void zchunked_wrapper<CTE>::assign(xarray<value_type>&& rhs)
    {
        assign_impl(std::move(rhs));
    }

    template <class CTE>
    auto zchunked_wrapper<CTE>::clone() const -> self_type*
    {
        return new self_type(*this);
    }

    template <class CTE>
    zarray_impl* zchunked_wrapper<CTE>::strided_view(xstrided_slice_vector& slices)
    {
        auto e = xt::strided_view(m_chunked_array, slices);
        return detail::build_zarray(std::move(e));
    }

    template <class CTE>
    auto zchunked_wrapper<CTE>::get_metadata() const -> const nlohmann::json&
    {
        return m_metadata;
    }

    template <class CTE>
    void zchunked_wrapper<CTE>::set_metadata(const nlohmann::json& metadata)
    {
        m_metadata = metadata;
    }

    template <class CTE>
    std::size_t zchunked_wrapper<CTE>::dimension() const
    {
        return m_chunked_array.dimension();
    }

    template <class CTE>
    auto zchunked_wrapper<CTE>::shape() const -> const shape_type&
    {
        return m_chunked_array.shape();
    }

    template <class CTE>
    void zchunked_wrapper<CTE>::resize(const shape_type&)
    {
        // This function is called because zarray (and therefore
        // all the wrappers) implements the container semantic,
        // whatever it wraps (and it cannot be different, since
        // the type of the wrappee is erased).
        // We can see this as a mapping container_semantic => chunked_semantic,
        // meaning this function call must be authorized, and must not do
        // anything.
    }

    template <class CTE>
    void zchunked_wrapper<CTE>::resize(shape_type&&)
    {
        // See comment above.
    }

    template <class CTE>
    bool zchunked_wrapper<CTE>::broadcast_shape(shape_type& shape, bool reuse_cache) const
    {
        return m_chunked_array.broadcast_shape(shape, reuse_cache);
    }

    template <class CTE>
    auto zchunked_wrapper<CTE>::chunk_shape() const -> const shape_type&
    {
        return m_chunk_shape;
    }

    template <class CTE>
    inline void zchunked_wrapper<CTE>::compute_cache() const
    {
        if (!m_cache_initialized)
        {
            m_cache = m_chunked_array;
            m_cache_initialized = true;
        }
    }

    template <class CTE>
    template <class CT>
    inline detail::enable_const_t<CT> zchunked_wrapper<CTE>::assign_impl(xarray<value_type>&&)
    {
        throw std::runtime_error("const array is not assignable");
    }

    template <class CTE>
    template <class CT>
    inline detail::disable_const_t<CT> zchunked_wrapper<CTE>::assign_impl(xarray<value_type>&& rhs)
    {
        //throw std::runtime_error("this should work");
        m_chunked_array = rhs;
    }
}

#endif
