#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void updateEvolutionPaths(int numberOfVariables, //read
                                   __global float* artmp, //set
                                   __global float* invsqrtC2array, //read
                                   __global float* xmean, //read
                                   __global float* xold, //read
                                   __global float* ps, //set
                                   float sigma,
                                   float cs,
                                   float mueff
                                   )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId > numberOfVariables){
        return;
    }
    
    //Cumulation: Update evolution paths
    artmp[globalId] = 0;
    for (int j = 0; j < numberOfVariables; j++) {
        artmp[globalId] += invsqrtC2array[(numberOfVariables * globalId) + j] * (xmean[j] - xold[j]) / sigma;
    }
    
    //Cumulation for sigma (ps)
    ps[globalId] = (1 - cs) * ps[globalId] + sqrt(cs * (2 - cs) * mueff) * artmp[globalId];
    
}