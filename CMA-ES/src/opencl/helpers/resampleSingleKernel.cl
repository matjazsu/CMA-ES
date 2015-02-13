#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Number of kernels = populationSize */
__kernel void resampleSingle(int N,
                             __global float* upperLimit, //read
                             __global float* lowerLimit, //read
                             __global float* arx2Array, //set
                             int iNK //read
                             )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId >= N)
    {
        return;
    }
    
    if (arx2Array[(N * iNK) + globalId] > upperLimit[globalId])
    {
        arx2Array[(N * iNK) + globalId] = upperLimit[globalId];
    }
    else if (arx2Array[(N * iNK) + globalId] < lowerLimit[globalId])
    {
        arx2Array[(N * iNK) + globalId] = lowerLimit[globalId];
    }
    
}