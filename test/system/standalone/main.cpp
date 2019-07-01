/* Copyright 2019 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#if defined( CUPLA_ACC_CpuOmp2Blocks  )
#   include <cupla/standalone/CpuOmp2Blocks.hpp>
#elif defined( CUPLA_ACC_CpuOmp2Threads  )
#   include <cupla/standalone/CpuOmp2Threads.hpp>
#elif defined( CUPLA_ACC_CpuSerial )
#   include <cupla/standalone/CpuSerial.hpp>
#elif defined( CUPLA_ACC_CpuTbbBlocks  )
#   include <cupla/standalone/CpuTbbBlocks.hpp>
#elif defined( CUPLA_ACC_CpuThreads  )
#   include <cupla/standalone/CpuThreads.hpp>
#elif defined( CUPLA_ACC_GpuCudaRt  )
#   include <cupla/standalone/GpuCudaRt.hpp>
#elif defined( CUPLA_ACC_GpuHipRt  )
#   include <cupla/standalone/GpuHipRt.hpp>
#endif

extern void callIncrementKernel(int* pr_d);

int main()
{
    int res_ptr_h;
    int *res_ptr_d;
    cudaMalloc( (void**)&res_ptr_d, sizeof( int ) );

    // reset result to zero
    cuplaMemset( res_ptr_d, 0, sizeof( int ) );

    // increment 42 times
    callIncrementKernel(res_ptr_d);

    cudaMemcpy(&res_ptr_h, res_ptr_d, sizeof( int ), cudaMemcpyDeviceToHost);

    return res_ptr_h != 42;
}
