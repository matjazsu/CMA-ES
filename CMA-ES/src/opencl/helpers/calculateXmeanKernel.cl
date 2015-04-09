#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void calculateXmean(int populationSize, //read
                             int numberOfVariables, //read
                             __global int* arindex, //read
                             __global float* arx2Array, //read
                             __global float* weights, //read
                             __global float* xmean,
                             __global float* xold //set
                             )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId > numberOfVariables){
        return;
    }
    
    __private float ratio = populationSize/2;
    __private int mu = floor(ratio);
    
    // calculate xmean and BDz~N(0,C)
    xold[globalId] = xmean[globalId];
    xmean[globalId] = 0;
    
    for(int iNk = 0; iNk < mu; iNk++){
        xmean[globalId] += weights[iNk] * arx2Array[(numberOfVariables * arindex[iNk]) + globalId];
    }
}