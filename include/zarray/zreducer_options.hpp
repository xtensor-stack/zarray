#ifndef XTENSOR_ZREDUCER_OPTIONS_HPP
#define XTENSOR_ZREDUCER_OPTIONS_HPP

#include "xtensor/xcontainer.hpp"
#include "xtensor/xscalar.hpp"
#include "xtensor/xreducer.hpp"


namespace xt
{
    template <class CTE>
    class zscalar_wrapper;

    template <class F, class CT>
    class zreducer;

    class zreducer_options{
    public:


        template <class X>
        struct initial_tester : std::false_type {};

        template <class X>
        struct initial_tester<xinitial<X>> : std::true_type {};

        // Workaround for Apple because tuple_cat is buggy!
        template <class X>
        struct initial_tester<const xinitial<X>> : std::true_type {};



        template <class F, class CT>
        friend class zreducer;

    private:
        template<class T>
        using make_scalar_wrapper_t = zscalar_wrapper<xscalar<T>>;

        template<class EVS>
        void init_from_tuple(EVS && options);

    public:


        zreducer_options(const zreducer_options & ) = default;
        zreducer_options(zreducer_options && ) = default;
        zreducer_options & operator= ( const zreducer_options & ) = default;


        zreducer_options();


        template <class EVS,
            XTL_REQUIRES(
                is_reducer_options<std::decay_t<EVS>>
            )
        >
        zreducer_options(EVS && options)
        :   m_axes(),
            m_keep_dims(),
            m_lazy(),
            m_inital_value()
        {
            this->init_from_tuple(std::forward<EVS>(options));
        }


        template <class A, class EVS = DEFAULT_STRATEGY_REDUCERS,
            XTL_REQUIRES(
                xtl::negation<is_reducer_options<A>>,
                is_reducer_options<EVS>,
                xtl::negation<std::is_same<std::decay_t<A>, zreducer_options>>
            )
        >
        zreducer_options(A&& axes, EVS && options = EVS())
        :   m_axes(axes.begin(), axes.end()),
            m_keep_dims(),
            m_lazy(),
            m_inital_value()
        {

            this->init_from_tuple(std::forward<EVS>(options));
        }

        template <class I, std::size_t N,  class EVS = DEFAULT_STRATEGY_REDUCERS,
            XTL_REQUIRES(is_reducer_options<EVS>)
        >
        zreducer_options( const I(&axes)[N], EVS && options = EVS())
        :   m_axes(axes, axes+N),
            m_keep_dims(),
            m_lazy(),
            m_inital_value()
        {

            this->init_from_tuple(std::forward<EVS>(options));
        }

        template<class T>
        bool can_get_inital_value() const;

        template<class T>
        T get_inital_value() const;

        bool has_initial_value() const;

        bool keep_dims() const;

        const auto & axes() const;

        bool is_lazy()const;


        template<class T, class F>
        void visit_reducer_options(const bool force_lazy, F && f)const;

        template<class F>
        void visit_reducer_options( F && f) const;

    private:


        template<class F, class O>
        void handle_eval_strategie(F && f, const bool force_lazy, O && options) const;

        template<class F, class O>
        void handle_keep_dim(F && f, O && options) const;

        template<class F, class O>
        void run(F && f, O && options) const;

        svector<std::size_t> m_axes;
        bool m_keep_dims;
        bool m_lazy;

        // todo, replace this with an actual zarray itself
        // (atm this is not possible since this would lead to cyclic
        // import / incomplete types)
        std::shared_ptr<zarray_impl> m_inital_value;
    };

    inline zreducer_options::zreducer_options()
    :   m_axes(),
        m_keep_dims(false),
        m_lazy(true),
        m_inital_value()
    {}



    template<class T>
    inline bool zreducer_options::can_get_inital_value() const
    {

        return bool(m_inital_value) && dynamic_cast<   make_scalar_wrapper_t<T> *>(m_inital_value.get());
    }

    template<class T>
    inline T zreducer_options::get_inital_value() const
    {
        auto wrapper =  dynamic_cast<make_scalar_wrapper_t<T> *>(m_inital_value.get());
        return *(wrapper->get_array().begin());
    }

    inline bool zreducer_options::zreducer_options::has_initial_value() const
    {
        return bool(m_inital_value);
    }

    inline bool zreducer_options::keep_dims() const
    {
        return m_keep_dims;
    }

    inline const auto & zreducer_options::axes() const{
        return m_axes;
    }

    inline bool zreducer_options::is_lazy()const{
        return m_lazy;
    }

    // TODO consider to make this private
    // call the functor with the
    // matching xt::reducer_options
    // - we need to be able to enforce the laziness (for chunked arrays)
    // - this is called when we already dispatched the type of the potential
    //   inital value. Ie the type of the inital value must match T
    template<class T, class F>
    inline void zreducer_options::visit_reducer_options(const bool force_lazy, F && f)const
    {
        if(this->has_initial_value())
        {
            this->handle_eval_strategie(std::forward<F>(f),force_lazy, initial(this->get_inital_value<T>()));
        }
        else
        {
            this->handle_eval_strategie(std::forward<F>(f), force_lazy, std::tuple<>());
        }
    }

    // this overload is called in the result index dispatcher where
    // we do not have access to the result array yet. Since we abuse
    // the result arrays in the dispatcher to hold the initial value, we do not
    // have access to the dispatched type of the initial value.
    // Hence we generate the options object without any inital value
    template<class F>
    inline void zreducer_options::visit_reducer_options( F && f) const
    {
        this->handle_eval_strategie(std::forward<F>(f), false /*force lazy*/, std::tuple<>());
    }

    template<class EVS>
    inline void zreducer_options::init_from_tuple(EVS && options)
    {
        using tuple_type = std::decay_t<EVS>;
        m_lazy = tuple_idx_of<xt::evaluation_strategy::immediate_type, tuple_type>::value == -1;
        m_keep_dims = tuple_idx_of<xt::keep_dims_type, tuple_type>::value != -1;

        static constexpr std::size_t initial_val_idx = xtl::mpl::find_if<initial_tester, tuple_type>::value;

        xtl::mpl::static_if<initial_val_idx != std::tuple_size<tuple_type>::value>([this, &options](auto no_compile)
        {
            // use no_compile to prevent compilation if initial_val_idx is out of bounds!
            auto initial_value = no_compile(std::get<initial_val_idx != std::tuple_size<tuple_type>::value ? initial_val_idx : 0>(options)).value();
            using inital_value_type = std::decay_t<decltype(initial_value)>;
            m_inital_value = std::make_shared<  make_scalar_wrapper_t<inital_value_type>  >(xscalar<inital_value_type>(initial_value));
        },[](auto /*np_compile*/){});
    }

    template<class F, class O>
    inline void zreducer_options::handle_eval_strategie(F && f, const bool force_lazy, O && options) const
    {
        if(force_lazy || m_lazy)
        {
            this->handle_keep_dim(std::forward<F>(f),
                std::tuple_cat(std::forward<O>(options),std::tuple<xt::evaluation_strategy::lazy_type>()));
        }
        else
        {
            this->handle_keep_dim(std::forward<F>(f),
                std::tuple_cat(std::forward<O>(options), std::tuple<xt::evaluation_strategy::immediate_type>()));
        }
    }

    template<class F, class O>
    inline void zreducer_options::handle_keep_dim(F && f, O && options) const
    {
        if(m_keep_dims)
        {
            this->run(std::forward<F>(f), std::tuple_cat(std::forward<O>(options), std::tuple<keep_dims_type>()));
        }
        else
        {
            this->run(std::forward<F>(f), std::forward<O>(options));
        }
    }

    template<class F, class O>
    inline void zreducer_options::run(F && f, O && options) const
    {
        f(m_axes, std::forward<O>(options));
    }
}

#endif