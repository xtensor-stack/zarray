#include "zarray/zarray.hpp"
#include "test_init.hpp"

namespace xt
{
    void initialize_dispatchers()
    {
        init_zassign_dispatchers<void>();
    }
}
