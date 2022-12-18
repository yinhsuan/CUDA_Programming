#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int mandel(float c_re, float c_im, int maxIteration)
{
	float z_re = c_re, z_im = c_im;
	int i;
	for (i = 0; i < maxIteration; ++i)
	{
		if (z_re * z_re + z_im * z_im > 4.f)
		break;

		float new_re = z_re * z_re - z_im * z_im;
		float new_im = 2.f * z_re * z_im;
		z_re = c_re + new_re;
		z_im = c_im + new_im;
	}
	return i;
}


__global__ void mandelKernel(float lowerX, float lowerY, int resX, int resY, int maxIterations, float stepX, float stepY, int *device) {
    // To avoid error caused by the floating number, use the following pseudo code
    //

    // get the curreent thread location
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float x = lowerX + thisX * stepX;
    float y = lowerY + thisY * stepY;

    device[thisY*resX + thisX] = mandel(x, y, maxIterations);
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int threadsPerBlockX = 16;
    int threadsPerBlockY = 16;
    int size = resX * resY * sizeof(int);

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // allocate memory space
    int* host;
    int* device;
    host = (int*)malloc(size);
    cudaMalloc((void**)&device, size);

    // calculate
    dim3 block(threadsPerBlockX, threadsPerBlockY);
    dim3 grid(resX / block.x, resY / block.y);
    mandelKernel<<<grid, block>>>(lowerX, lowerY, resX, resY, maxIterations, stepX, stepY, *device)

    // copy
    cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost); //  device(gpu) -> host(cpu)
    memcpy(img, host, size); // host -> img

    // release the memory
    free(host);
    cudaFree(device);
}
