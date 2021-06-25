/***************************************************************************
* Copyright (c) Johan Mabille, Sylvain Corlay and Wolf Vollprecht          *
* Copyright (c) QuantStack                                                 *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_ZARRAY_BUFFER_HANDLER_HPP
#define XTENSOR_ZARRAY_BUFFER_HANDLER_HPP

#include <array>
#include <set>
#include <map>
#include <memory>

namespace xt
{
    // forward declare
    class zarray_impl;


    class zarray_buffer_handler
    {
    public:

        // buffer index is pair of typen index and buffer index for that type

        zarray_buffer_handler(zarray_impl & res)
        :   m_res(res),
            m_buffers(),
            m_free_buffers()
        {
            this->mark_as_free(&res);
        }

        auto get_free_buffer(const std::size_t type_index)
        {
            auto r = m_free_buffers.find(type_index);
            if(r == m_free_buffers.end() || r->second.empty())
            {
                // make new buffer
                auto buffer_ptr = zarray_impl_register::get(type_index).clone();
                m_buffers.emplace_back(buffer_ptr);
                return buffer_ptr;
            }
            else
            {
                // remove from free set
                auto iter = r->second.begin();
                auto buffer_ptr = *iter;
                r->second.erase(iter);

                return buffer_ptr;
            }
        };



        void mark_as_free(const zarray_impl * buffer_ptr)
        {

            m_free_buffers[buffer_ptr->get_class_index()].insert(
                const_cast<zarray_impl *>(buffer_ptr)
            );
        }


    private:

        // the "outer" result which is passed to
        // the root of the expression tree which
        // has a special role
        zarray_impl & m_res;

        // a vector of buffers since an arbitrary number of temps can be needed
        std::vector<std::unique_ptr<zarray_impl>>  m_buffers;


        std::map<std::size_t, std::set<zarray_impl * > >  m_free_buffers;
    };
}

#endif