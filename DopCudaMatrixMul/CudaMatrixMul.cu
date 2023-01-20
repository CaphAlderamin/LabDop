
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <stdio.h>
#include <chrono>
#include <iostream>

using namespace std;

const unsigned int threadsPerBlock_ = 16;
const unsigned int dim = 1536;
__int64 A[dim][dim];
__int64 B[dim][dim];
__int64 C[dim][dim];
__int64 D[dim][dim];

__global__ void mulKernel(__int64* A, __int64* B, __int64* C, unsigned int dim)
{
    __shared__ __int64 Ads[threadsPerBlock_][threadsPerBlock_];
    __shared__ __int64 Bds[threadsPerBlock_][threadsPerBlock_];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Col = bx * threadsPerBlock_ + tx;
    int Row = by * threadsPerBlock_ + ty;

    __int64 Pervalue = 0;

    for (int i = 0; i < dim / threadsPerBlock_; i++) // multiply blocks
    {
        Ads[ty][tx] = A[Row * dim + (i * threadsPerBlock_ + tx)];
        Bds[ty][tx] = B[Col + (i * threadsPerBlock_ + ty) * dim];
        __syncthreads();


        for (int k = 0; k < threadsPerBlock_; k++) // multiply threadsPerBlock_
            Pervalue += Ads[ty][k] * Bds[k][tx];
        __syncthreads();
    }

    C[Row * dim + Col] = Pervalue;
}

    

int mulWithCUDA()
{
    const unsigned int printDim = 12;

    cudaError_t cudaStatus;
    srand(time(NULL));

    // Matrix initialize
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }
    }

    // ============================ GPU CALCULATION ============================
    __int64* dev_A;
    __int64* dev_B,
    __int64* dev_C;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_A, dim * dim * sizeof(__int64));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc A failed!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_B, dim * dim * sizeof(__int64));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc B failed!\n");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_C, dim * dim * sizeof(__int64));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc C failed!\n");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_A, A, dim * dim * sizeof(__int64), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy A failed!\n");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_B, B, dim * dim * sizeof(__int64), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy B failed!\n");
        goto Error;
    }

    dim3 threadsPerBlock(threadsPerBlock_, threadsPerBlock_);
    dim3 numBlocks(dim / threadsPerBlock.x, dim / threadsPerBlock.y);

    auto start = chrono::high_resolution_clock::now(); // time start

    // Launch a kernel on the GPU.
    mulKernel <<< numBlocks, threadsPerBlock >>> (dev_A, dev_B, dev_C, dim);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaThreadSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    auto end = chrono::high_resolution_clock::now(); // time end
    chrono::duration<float> durTime = end - start;
    cout << "GPU calculating time: " << durTime.count() << endl;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, dev_C, dim * dim * sizeof(__int64), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy C failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    // =========================================================================
    
    // ============================ CPU CALCULATION ============================
    start = chrono::high_resolution_clock::now(); // time start
   
    for (size_t i = 0; i < dim; ++i)
    {
        for (size_t j = 0; j < dim; ++j)
        {
            D[i][j] = 0;

            for (size_t k = 0; k < dim; ++k)
            {
                D[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    end = chrono::high_resolution_clock::now(); // time end
    durTime = end - start;
    cout << "CPU calculating time: " << durTime.count() << endl;
    // =========================================================================
   
    // Print all matrix
    cout << endl;
    /*for (int i = 0; i < printDim; i++)
    {
        for (int j = 0; j < printDim; j++)
        {
            cout << A[i][j] << " ";
        }
        cout << "   ";
        for (int j = 0; j < printDim; j++)
        {
            cout << B[i][j] << " ";
        }
        cout << "   ";
        for (int j = 0; j < printDim; j++)
        {
            cout << C[i][j] << " ";
        }
        cout << "   ";
        for (int j = 0; j < printDim; j++)
        {
            cout << D[i][j] << " ";
        }
        cout << endl;
    }*/

    // ================== Сравнение полученных матриц ==================
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            if (C[i][j] != D[i][j])
            {
                cout << "GPUResult[" << i << "][" << j << "] = " << C[i][j] << "  != " << " CPUResult[" << i << "][" << j << "] = " << D[i][j] << endl;
                goto Error;
            }
        }
    }
    //==================================================================

Error:
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void main()
{
    int repeats = 10;
    for (int i = 0; i < repeats; i++)
    {
        mulWithCUDA();
    }
}