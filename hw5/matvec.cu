#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <iostream>
void matvec(double* y, const double* A, const double* x, long N){
  double sum;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++){
  	sum=0.0;
	for (long j=0;j<N;j++){
		sum += A[i+j*N]*x[j];	
	}
	y[i]=sum;
  } 
}

double cuda_error(const double* y_ref, const double* y, long N){
  double sum=0.0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++){
	sum+=(y_ref[i]-y[i])*(y_ref[i]-y[i]);
  }
  double error=sqrt(sum);
  return error;
}
void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

// Warp divergence
__global__ void reduction_kernel0(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x %   2 == 0) smem[threadIdx.x] += smem[threadIdx.x + 1];
  __syncthreads();
  if (threadIdx.x %   4 == 0) smem[threadIdx.x] += smem[threadIdx.x + 2];
  __syncthreads();
  if (threadIdx.x %   8 == 0) smem[threadIdx.x] += smem[threadIdx.x + 4];
  __syncthreads();
  if (threadIdx.x %  16 == 0) smem[threadIdx.x] += smem[threadIdx.x + 8];
  __syncthreads();
  if (threadIdx.x %  32 == 0) smem[threadIdx.x] += smem[threadIdx.x + 16];
  __syncthreads();
  if (threadIdx.x %  64 == 0) smem[threadIdx.x] += smem[threadIdx.x + 32];
  __syncthreads();
  if (threadIdx.x % 128 == 0) smem[threadIdx.x] += smem[threadIdx.x + 64];
  __syncthreads();
  if (threadIdx.x % 256 == 0) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x % 512 == 0) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + 512];
}

// Shared memory bank conflicts
__global__ void reduction_kernel1(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x *   2] += smem[threadIdx.x *   2 +   1];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x *   4] += smem[threadIdx.x *   4 +   2];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x *   8] += smem[threadIdx.x *   8 +   4];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x *  16] += smem[threadIdx.x *  16 +   8];
  __syncthreads();
  if (threadIdx.x <  32) smem[threadIdx.x *  32] += smem[threadIdx.x *  32 +  16];
  __syncwarp();
  if (threadIdx.x <  16) smem[threadIdx.x *  64] += smem[threadIdx.x *  64 +  32];
  __syncwarp();
  if (threadIdx.x <   8) smem[threadIdx.x * 128] += smem[threadIdx.x * 128 +  64];
  __syncwarp();
  if (threadIdx.x <   4) smem[threadIdx.x * 256] += smem[threadIdx.x * 256 + 128];
  __syncwarp();
  if (threadIdx.x <   2) smem[threadIdx.x * 512] += smem[threadIdx.x * 512 + 256];
  __syncwarp();
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[512];
}

__global__ void innerprod_kernel2(double* sum, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx]*b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void reduction_kernel2(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void matvec_kernel(double *y, const double *A, const double *x, long N){
	int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	if (idx<N){
		double sum=0;
		for (long j=0;j<N;j++){
			sum+=A[idx+j*N]*x[j];
		}
		y[idx]=sum;
	}

}

int main() {
  long N = 2048;

  double *x;
  double *y; // we want to compute the inner product of x and y
  double *A;
  double *y_ref;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&y_ref, N * sizeof(double));
  cudaMallocHost((void**)&A, N *N* sizeof(double));
  std::cout<<"N="<<N<<std::endl;


#pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) x[i] = 1.0/(i+1);
  for (long i = 0; i < N*N; i++) A[i] = 1.0/(i+1);
  for (long i = 0; i < N; i++) y[i] = 0;//we want to compute y=Ax
  for (long i = 0; i < N; i++) y_ref[i] = 0;

  double sum;
  double tt = omp_get_wtime();
  matvec(y_ref, A, x, N);
  printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *z_d, *A_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&A_d, N*N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
 
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&z_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(A_d, A, N*N*sizeof(double), cudaMemcpyHostToDevice);
  
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  matvec_kernel<<<Nb,BLOCK_SIZE>>>(y_d,A_d,x_d,N);
  cudaMemcpyAsync(y, y_d, N*sizeof(double), cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %f\n",cuda_error(y_ref, y, N));
  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);
  cudaFree(A_d);
  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(A);
  return 0;
}

