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


__global__ void mandelKernel(float lowerX, float lowerY, int resX, int resY, int maxIterations, float stepX, float stepY, int *device, int groupX, int groupY) {
    // To avoid error caused by the floating number, use the following pseudo code
    //

    // get the curreent thread location
    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * groupX;
    int thisY = (blockIdx.y * blockDim.y + threadIdx.y) * groupY;
    // if (thisX >= resX || thisY >= resY) return;

    int i, j;
    float x, y;
    for (j=thisY; j<thisY+groupY; j++) {
        // if (j >= resY) return;
        for (i=thisX; i<thisX+groupX; i++) {
            // if (i >= resX) continue;
            x = lowerX + i * stepX;
            y = lowerY + j * stepY;
            device[j*resX + i] = mandel(x, y, maxIterations);
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE(float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int threadsPerBlock = 16;
    int groupX = 4;
    int groupY = 4;
    int size = resX * resY * sizeof(int);

    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    // allocate memory space
    int* host;
    int* device;
    size_t pitch;
    cudaHostAlloc((void**)&host, size, cudaHostAllocMapped);
    cudaMallocPitch((void**)&device, &pitch, resX * sizeof(int), resY);

    // calculate
    dim3 block((int) ceil(threadsPerBlock / groupX), (int) ceil(threadsPerBlock / groupY));
    dim3 grid((int) ceil(resX / threadsPerBlock), (int) ceil(resY / threadsPerBlock));
    mandelKernel<<<grid, block>>>(lowerX, lowerY, resX, resY, maxIterations, stepX, stepY, device, groupX, groupY);

    // cudaDeviceSynchronize();

    // copy
    cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost); //  device(gpu) -> host(cpu)
    memcpy(img, host, size); // host -> img

    // release the memory
    cudaFreeHost(host);
    cudaFree(device);
}
