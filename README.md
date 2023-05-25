# CUDA_Programming

### Part 1: Paralleling Fractal Generation with CUDA


### Q1 What are the pros and cons of the three methods? Give an assumption about their performances.
- Method 1:
    - Pros: 
        - **`Higher parallelism`**: Each thread only needs to handle the calculation of one pixel. Therefore, there would be relatively many warps for SM to switch from.
        - **`Fewer memory usage`**: Because we use "malloc" to allocate the host memory, we adopt `pageable memory`. As a result, we can save the `pinned memory space`.
            > 1. Pageable Memory: the memory space allocate by malloc() is managed  by "page table". Therefore, the data in the memory is likely to be written into the "swap space".
            > 2. For the purpose of not letting the data be written from the pageable memory to the swap space, the CUDA API would implicitly copy the space allocated by malloc() to `"pinned memory" (page-lock)`. After that, the gpu can then copy the data on host's pinned memory to devide memory using DMA (Direct Memory Access).
            > 3. The data transfer between host and CUDA device requires `page-locked memory` on host. Because `higher bandwidth` is possible between the host and the device when using page-locked (or “pinned”) memory. Therefore, the transmission could be faster.
    - Cons: 
        - **`Different loading of each thread`**: Each thread will handle the calculation of only one pixel. Therefore, the calculation will varies between each thread. The finished threads need to wait for the unfinished threads.
        - **`Extra time for copying data`**: According to `Quote 3` mentioned in Pros, the CUDA API would implicitly copy the space allocated by malloc() to `"pinned memory" (page-lock)`. Therefore, there is one more copy operation and it require additional the time for the transmission.

- Method 2:
    - Pros: 
        - **`Higher parallelism`**: Each thread only needs to handle the calculation of one pixel. Therefore, there would be relatively many warps for SM to switch from.
        - **`Reduce the time for copying data`**: Because we use “cudaHostAlloc” to allocate the host memory, we can directly put the data that will be transmitted to device on the host's `pinned memory`. In this way, we can save the time for copying data from pageable memory to temporary pinned memory.
        - **`Higher data access rate in device`**: From the [cuda document](http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html#g80d689bc903792f906e49be4a0b6d8db), we can see that cudaMallocPitch() will pad the allocation to ensure the `alignment` requirements. Therefore, it can enhance the data access rate for gpu.
    - Cons: 
        - **`Different loading of each thread`**: Each thread will handle the calculation of only one pixel. Therefore, the calculation will varies between each thread. The finished threads need to wait for the unfinished threads.
        - **`Higher memory usage`**: Because we use “cudaHostAlloc” to allocate the host memory, we can directly put the data on the host’s pinned memory. Therefore, the data cannot be written to the swap space and it requires `pinned memory space` in advance.
        - **`Lower memory utilization`**: Because cudaMallocPitch() will pad the allocation to ensure the alignment requirements, some memory is only used for alignment without actual utilization. Therefore, those memory spaces are wasted.

- Method 3:
    - Pros: 
        - **`Equalize the loading of each thread`**: Each thread will handle the calculation of a group of pixels. Therefore, the calculation will be more equal for each thread. This is because the calculation can be shared by more threads instead of only a few threads.
        - **`Reduce the time for copying data`**: Because we use “cudaHostAlloc” to allocate the host memory, we can directly put the data on the host's `pinned memory`. In this way, we can save the time for copying data from pageable memory to temporary pinned memory.
        - **`Higher data access rate in device`**: From the [cuda document](http://horacio9573.no-ip.org/cuda/group__CUDART__MEMORY_g80d689bc903792f906e49be4a0b6d8db.html#g80d689bc903792f906e49be4a0b6d8db), we can see that cudaMallocPitch() will pad the allocation to ensure the `alignment` requirements. Therefore, it can enhance the data access rate for gpu.
    - Cons: 
        - **`Reduce the utilization rate of GPU`**: Each thread will handle the calculation of a group of pixels. Therefore, it requires less threads for the calculation. In other words, the number of warp reduces. Once there is no enough warp for SM switch from, the utilization rate of GPU reducws.
            > When a warp is not excecuting, SM can switch in another available warp to execute that warp. In the meanwhile, the switching operation between warps does not cost anything because the staus of warps is already stored in SM.
        - **`Higher memory usage`**: Because we use “cudaHostAlloc” to allocate the host memory, we can directly put the data on the host’s pinned memory. Therefore, the data cannot be written to the swap space and it requires `pinned memory space` in advance.
        - **`Lower memory utilization`**: Because cudaMallocPitch() will pad the allocation to ensure the alignment requirements, some memory is only used for alignment without actual utilization. Therefore, those memory spaces are wasted.

- My Assumption:
    - I think the performance of these three methods would be: 
        - Method 2 > Method 1 > Method 3
    - This is because Method 2 have better parallelism due to its address alignment. And also, it has better GPU utilization rate due to the large amount of threads. As for its low memory utilization happends only when the the program requires a large amount of pinned memory space. And this situation does not happen here. Therefore, the performance of Method 2 might be the best.
    - Because Method 3 have each thread handles the calculation of a group of pixels, it contains the serial part in each thread. And also, Method 3 requires less threads to do the calculation, so the utilization rate of GPU might not be as well as Method 1 and Method 2. Therefore, the performance of Method 3 might be the worst.
    - Method 1 has better GPU utilization rate due to the large amount of threads. Even though it need to copy the data from pageable memory to pinned memory, the memory size of the program is not very large here. Thus, the performance of Method 1 might be in between.
    - [reference1](https://kaibaoom.tw/2020/07/21/cuda-four-memory-access/)
    - [reference2](https://leimao.github.io/blog/Page-Locked-Host-Memory-Data-Transfer/#:~:text=Pageable%20Memory%20and%20Page%2DLocked,locked%20memory%20or%20pinned%20memory.)
    - [reference3](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
    - [reference4](https://blog.51cto.com/u_15060511/3807762)
    - [reference5](https://icl.utk.edu/~mgates3/docs/cuda.html)




### Q2 How are the performances of the three methods? Plot a chart to show the differences among the three methods
- for VIEW 1 and VIEW 2, and
- for different maxIteration (1000, 10000, and 100000).

![](https://i.imgur.com/ajAMI79.png)
![](https://i.imgur.com/zJliu9j.png)




### Q3 Explain the performance differences thoroughly based on your experimental results. Does the results match your assumption? Why or why not.

> The brightness of each pixel is proportional to the computational cost of determining whether the value is contained in the Mandelbrot set
#### First of all, the performance of View 2 is much better than View 1. The main difference of View 2 and View 1 is that the white part is more uniform in View 2. Therefore, the workload to calculate each pixel in View 2 is more uniform. This can reduce the chance to cause the brance divergence.

#### Secondly, we can see that the performance of Method 3 is the worst among the three methods. This might because Method 3 have each thread handles the calculation of a group of pixels. In addition, Method 3 requires less threads to do the calculation, so the utilization rate of GPU might not be as well as Method 1 and Method 2.

#### Last but not least, because I used Mapped memory (zero-copy memory) in Method 2, I can avoid explicit data transfers between host and device and save the transfer time. However, as it is mapped into the device address space, the data will not be copied into the device memory. Therefore, transfer will happen during execution. As a result, it will increase the processing time. So, the execution time of Method 2 is slightly larger than Method 1.
[reference6](https://medium.com/analytics-vidhya/cuda-memory-model-823f02cef0bf)

### Q4 Can we do even better? Think a better approach and explain it. Implement your method in kernel4.cu

#### We can directly copy the data from the device to the image, instead of copying the data through the memory space on the host. (My kernel4.cu is the extension from kernel1.cu)
- Save the memory space on the host
- Save the time for copying the data (from the device to the host & from the host to the image)
- Save the time for initializing and deleting the memory space on the host

#### The performance of Method 4 is better than the other three methods!

![](https://i.imgur.com/3hf0c1M.png)
![](https://i.imgur.com/4vUBRtF.png)

