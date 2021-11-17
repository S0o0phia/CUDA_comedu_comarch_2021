#include <stdio.h>
#include <time.h>

__global__ void gpu_loop()
{
	printf("GPU::This is iteration number %d\n", threadIdx.x);
}

__host__ void cpu_loop(int n)
{
	for(int i = 0; i < n; i++)
		printf("CPU::This is iteration number %d\n", i);
}

int main()
{
	int n, b;
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
	printf("n, b?\n");
	scanf("%d %d", &n, &b);

	cudaEventRecord(start,0);

	gpu_loop<<<b, n>>>();

	//commented out the functions which helps to calculate time
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    
	float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

	clock_t begin = clock();
    cpu_loop(n);
    clock_t end = clock();
    double time_spent = (double)1000 * (end - begin) / CLOCKS_PER_SEC;

	printf("GPU time: %f\n", et);
	printf("CPU time: %f", time_spent);
}
