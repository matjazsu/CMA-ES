#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

__kernel void updateDistributionHelper(int N,
                               __global float* artmp22array,
                               __global float* B2Array,
                               __global float* diagD,
                               __global float* invsqrtC2array,
                               __global float* C2Array,
                               __global float* offdiag
                               )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId > N){
        return;
    }
    
    for (int j = 0; j <= globalId; j++) {
        B2Array[(N * globalId) + j] = B2Array[(N * j) + globalId] = C2Array[(N * globalId) + j];
    }
    
    diagD[globalId] = sqrt(diagD[globalId]);
    
    for (int j = 0; j < N; j++) {
        artmp22array[(N * globalId) + j] = B2Array[(N * globalId) + j] * (1 / diagD[j]);
    }
    
    for (int j = 0; j < N; j++) {
        invsqrtC2array[(N * globalId) + j] = 0;
        for (int k = 0; k < N; k++) {
            invsqrtC2array[(N * globalId) + j] += artmp22array[(N * globalId) + k] * B2Array[(N * j) + k];
        }
    }
}