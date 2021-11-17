#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <random>

__device__ void swap_gpu(int* x, int* y)
{
	int tmp;

	tmp = *x;
	*x = *y;
	*y = tmp;
}

__global__ void doSomething_GPU(int* x, int I, int n)
{
	int tmp;
	int id = blockIdx.x;

	if(I == 0 && ((id * 2 + 1) < n)) {
		if(x[id * 2] > x[id * 2 + 1])
			swap_gpu(&x[id * 2], &x[id * 2 + 1]);
	}

	if(I == 1 && ((id * 2 + 2) < n)) {
		if(x[id * 2 + 1] > x[id * 2 + 2])
			swap_gpu(&x[id * 2 + 1], &x[id * 2 + 2]);
	}
}

__host__ void swap_cpu(int* x, int* y)
{
	int tmp;

	tmp = *x;
	*x = *y;
	*y = tmp;
}

__host__ void doSomething_CPU(int* arr, int n)
{
	bool done = false;

	while(!done) {
		done = true;
	
		for (int i = 1; i <= n - 2; i = i + 2) {
			if (arr[i] > arr[i + 1]) {
					swap_cpu(&arr[i], &arr[i + 1]);
					done = false;
				}
		}
  
		for (int i = 0; i <= n - 2; i = i + 2) {
			if (arr[i] > arr[i + 1]) {
				swap_cpu(&arr[i], &arr[i + 1]);
				done = false;
			}
		}
	}
}

int main()
{
	int *d;
	int n, i, a[50000], c[50000], g[50000];

	float time_gpu, time_cpu;
	cudaEvent_t start, stop;

	printf("N: ");
	scanf("%d", &n);

	printf("Target array is: ");
	for(i=0; i < n; i++) {
//		a[i] = rand() % 10000;
		scanf("%d", &a[i]);
		c[i] = a[i];
		printf("%d\t", c[i]);
	}

	cudaMalloc((void **) &d, n * sizeof(int));
	cudaMemcpy(d, a, n * sizeof(int), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	for(i = 0; i < n; i++)
		doSomething_GPU <<< 2, n / 2 >>> (d, i % 2, n);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time_gpu, start, stop);

	cudaMemcpy(g, d, n * sizeof(int), cudaMemcpyDeviceToHost);
	
	clock_t begin = clock();
    doSomething_CPU(c, n);
    clock_t end = clock();
    time_cpu = (double)(end - begin);

	printf("\nResult:\t");
	for(i=0; i<n; i++)	printf("%d\t",c[i]);

	printf("\nGPU time: %lf ms", time_gpu);
	printf("\nCPU time: %lf ms \n", time_cpu);

	cudaFree(d);
	return 0;
}
