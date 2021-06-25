/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZARRAY_BUFFER_MANAGER_HPP
#define XTENSOR_ZARRAY_BUFFER_MANAGER_HPP

#include <array>

namespace xt
{
    // forward declare
    class zarray_impl;


    class zarray_buffer_manager
    {
    public:

        // buffer index is pair of typen index and buffer index for that type
        using buffer_index_type = std::pair<std::size_t, std::size_t>;

        zarray_buffer_manager(zarray_impl & res)
        :   m_res(res),
            m_buffer(),
            m_is_free{true,true}
        {
        }

        auto get_free_buffer_index() const
        {
            return m_is_free[0] ? 0 : 1;
        };

        auto get_free_buffer_index_and_mark_as_used() const
        {
            auto index = this->get_free_buffer_index();
            this->mark_as_used(index);
        };

        auto & get_and_mark_as_used()
        {
            return this->get_buffer(this->get_free_buffer_index_and_mark_as_used());
        }

        void mark_as_free(const bool index)
        {
            m_is_free[static_cast<std::size_t>(index)] = true;
        }
        void mark_as_used(const bool index)
        {
            m_is_free[static_cast<std::size_t>(index)] = false;
        }

        zarray_impl & get_buffer(const bool index)
        {
            if( index == false)
            {
                return m_res;
            }
            else
            {
                if(!bool(m_buffer))
                {
                    m_buffer.reset(m_res.clone());
                }
                return &(m_buffer->get());
            }
        }

        // void assign_to_result(zarray_impl & data)
        // {
        //     if(bool(m_buffer) && &data == m_buffers.get())
        //     {
        //         m_res = *m_buffer.get();
        //     }

        // }

        zarray_impl & result_buffer()
        {
            return m_res;
        }

    private:

        // the "outer" result which is passed to
        // the root of the expression tree which
        // has a special role
        zarray_impl & m_res;

        // a vector of buffers since an arbitrary number of temps can be needed
        std::vector<std::unique_ptr<zarray_impl>>  m_buffers;


        // map from type-index to vector of buffer ptrs
        // ownership is managed by m_buffers
        // This map allows to access all buffers but also
        // "outer" result (m_res) in a uniform way
        std::map<std::size_t, std::vector<zarray_impl *>?
    };
}

#endif