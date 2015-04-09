#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void adaptCovarianceMatrix(int populationSize, //read
                                    int numberOfVariables, //read
                                    int counteval, //read
                                    float cmu, //read
                                    float c1, //read
                                    float cc, //read
                                    float sigma, //read
                                    float cs, //read
                                    float chiN, //read
                                    float mueff, //read
                                    __global float* psxps, //read
                                    __global float* xmean, //read
                                    __global float* xold, //read
                                    __global float* C2array, //set
                                    __global float* pc, //read
                                    __global float* weights, //read
                                    __global int* arindex, //read
                                    __global float* arx2array //read
                                    )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId > numberOfVariables){
        return;
    }
    
    //private variable
    __private int hsig = 0;
    __private float ration = populationSize/2;
    __private int mu = floor(ration);
    
    if ((sqrt(psxps[0]) / sqrt(1 - pow(1 - cs, 2 * counteval/populationSize)) / chiN) < (1,4 + 2 / (numberOfVariables + 1))){
        hsig = 1;
    }
    
    pc[globalId] = (1 - cc) * pc[globalId] + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean[globalId] - xold[globalId]) / sigma;
    
    for (int j = 0; j <= globalId; j++) {
        C2array[(numberOfVariables * globalId) + j] =  (1 - c1 - cmu)
        * C2array[(numberOfVariables * globalId) + j]
        + c1
        * (pc[globalId] * pc[j] + (1 - hsig) * cc * (2 - cc) * C2array[(numberOfVariables * globalId) + j]);
        for (int k = 0; k < mu; k++) {
            C2array[(numberOfVariables * globalId) + j] += cmu
            * weights[k]
            * (arx2array[(numberOfVariables * arindex[k]) + globalId] - xold[globalId])
            * (arx2array[(numberOfVariables * arindex[k]) + j] - xold[j]) / sigma
            / sigma;
        }
    }
}


