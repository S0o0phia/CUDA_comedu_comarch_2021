#include <stdio.h>
#include <time.h>

__global__ void matrixMulGPU( int * a, int * b, int * c, int N )
{
    int val = 0;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N)
    {
        for ( int k = 0; k < N; ++k )
            val += a[row * N + k] * b[k * N + col];
        c[row * N + col] = val;
    }
}

__host__ void matrixMulCPU( int * a, int * b, int * c, int N )
{
    int val = 0;

    for( int row = 0; row < N; ++row )
        for( int col = 0; col < N; ++col )
        {
            val = 0;
            for ( int k = 0; k < N; ++k )
                val += a[row * N + k] * b[k * N + col];
            c[row * N + col] = val;
        }
}

__host__ void printMatrix(int *mat, int N) {
	for( int row = 0; row < N; ++row ){
        for( int col = 0; col < N; ++col )
            printf("%d\t", mat[row * N + col]);
		printf("\n");
	}
}

int main()
{
    int N, size, *a, *b, *c_cpu, *c_gpu;

	printf("N?\n");
	scanf("%d", &N);

    size = N * N * sizeof (int); // Number of bytes of an N x N matrix

    // Allocate memory
    cudaMallocManaged (&a, size);
    cudaMallocManaged (&b, size);
    cudaMallocManaged (&c_cpu, size);
    cudaMallocManaged (&c_gpu, size);

    // Initialize memory
    for( int row = 0; row < N; ++row ){
        for( int col = 0; col < N; ++col )
        {
            a[row*N + col] = row;
            b[row*N + col] = col+2;
            c_cpu[row*N + col] = 0;
            c_gpu[row*N + col] = 0;
        }
	}

	printf("\nMatrix A:\n");
	printMatrix(a, N);

	printf("\nMatrix B:\n");
	printMatrix(b, N);

	printf("\n\n");

    dim3 threads_per_block (16, 16, 1); // A 16 x 16 block threads
    dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	cudaEventRecord(start,0);

    matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu, N );

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);	// Wait for the GPU to finish before proceeding
    
	float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
	
    // Call the CPU version to check our work
	clock_t begin = clock();
    matrixMulCPU( a, b, c_cpu, N );
    clock_t end = clock();
    double time_spent = (double)1000 * (end - begin) / CLOCKS_PER_SEC;

	printf("GPU time: %f\n", et);
	printf("CPU time: %f", time_spent);

    // Compare the two answers to make sure they are equal
    bool error = false;
    for( int row = 0; row < N && !error; ++row )
        for( int col = 0; col < N && !error; ++col )
            if (c_cpu[row * N + col] != c_gpu[row * N + col])
            {
                printf("FOUND ERROR at c[%d][%d]\n", row, col);
                error = true;
                break;
            }
    if (!error){
        printf("\nSuccess!\n");		
		printf("\nResult:\n");
		//printMatrix(c_gpu, N);
	}

    // Free all our allocated memory
    cudaFree(a); cudaFree(b);
    cudaFree( c_cpu ); cudaFree( c_gpu );
}