#include <cstdio>
#include <cstdlib>

#define BLOCK_SIZE 128

// Error handling macro
#define CUDA_ERR_CHECK(x) \
    do { cudaError_t err = x; if (err != cudaSuccess) { \
        fprintf (stderr, "Error \"%s\" at %s:%d \n", \
        cudaGetErrorString(err), \
        __FILE__, __LINE__); exit(-1); \
    }} while (0);

// Variable in GPU constant memory to hold the value of PI.
__constant__ float M_PI_GPU;

// TODO: Redefine the function to have both host and device
// implementations (see slide XXX of cuda_intro.pdf)
float cpu_kernel(float period, int i)
{
    return sinf(2.0f * float(M_PI) / period * float(i));
} 

// The kernel to be executed in many threads
__global__ void gpu_kernel(float period, float* result)
{
    // TODO: Calculate linear array index from thread and block index
    // int i = ...;
    
    // Do the calculations, corresponding to the thread
    result[i] = cpu_kernel(period, i);
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s <n>\n", argv[0]);
        printf("Where n must be a multiplier of %d\n", BLOCK_SIZE);
        return 0;
    }

    int n = atoi(argv[1]), nb = n * sizeof(float);
    printf("n = %d\n", n);
    if (n <= 0)
    {
        fprintf(stderr, "Invalid n: %d, must be positive\n", n);
        return 1;
    }
    if (n % BLOCK_SIZE)
    {
        fprintf(stderr, "Invalid n: %d, must be a multiplier of %d\n",
            n, BLOCK_SIZE);
        return 1;
    }

    float period = 256.0f;

    float* result = (float*)malloc(nb);

    // TODO: Allocate memory on GPU (see slide 47 of cuda_intro.pdf)
    float* resultDev = NULL;
    // CUDA_ERR_CHECK(cudaMalloc ...

	// TODO: Copy the value of PI to M_PI_GPU variable in GPU constant memory
	// (see cudaMemcpyToSymbol in CUDA API Reference manual).
    float m_pi = (float)M_PI;
    // CUDA_ERR_CHECK(cudaMemcpyToSymbol ...
    
    // TODO Set up the kernel launch configuration for n threads:
    // each block shall have BLOCK_SIZE threads, the number of blocks
    // shall be set to get n threads in total.
    // dim3 threads = ...
    // dim3 blocks  = ...

    // TODO Launch the kernel using the compute grid configuration above
    // gpu_kernel( ...
    
    // TODO Check the error from the kernel launch using cudaGetLastError()
    
    // TODO Wait for CUDA kernel to complete using cudaDeviceSynchronize()
    
    // TODO Copy GPU results back to CPU memory
    // CUDA_ERR_CHECK(cudaMemcpy ...

    // TODO Free GPU memory
    // CUDA_ERR_CHECK(cudaFree ...

    int imaxdiff = 0;
    float maxdiff = 0.0f;
    float maxdiff_good = 0.0f;
    float maxdiff_bad = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float gold = cpu_kernel(period, i);
        float diff = result[i] / gold;
        if (diff != diff) diff = 0;
        else diff = 1.0 - diff;
        if (diff > maxdiff)
        {
            maxdiff = diff;
            imaxdiff = i;
            maxdiff_good = gold;
            maxdiff_bad = result[i];
        }
    }

    printf("Max diff = %f%% @ i = %d: %f != %f\n",
        maxdiff * 100, imaxdiff, maxdiff_bad, maxdiff_good);

    return 0;
}

