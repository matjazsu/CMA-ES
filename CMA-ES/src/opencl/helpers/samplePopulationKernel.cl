#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Number of kernels = populationSize */
__kernel void samplePopulation(int N,
                               __global float* artmp, //read
                               __global float* B2Array, //read
                               __global float* arx2Array, //set
                               __global float* xmean, //read
                               float sigma, //read
                               int iNK //read
                               )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId >= N){
        return;
    }
    
    //private memory variable
    __private float sum = 0;
    
    for(int j = 0; j < N; j++){
        //write to private memory
        sum += B2Array[(N * globalId) + j] * artmp[j];
    }
    
    //write to global memory
    arx2Array[(N * iNK) + globalId] = (xmean[globalId] + sigma * sum);
}