#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void samplePopulation(int populationSize, //read
                               int numberOfVariables, //read
                               __global float* diagD, //read
                               __global float* B2Array, //read
                               __global float* arx2Array, //set
                               __global float* xmean, //read
                               __global float* randomArray, //read
                               float sigma //read
                               )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId > populationSize){
        return;
    }
    
    //private memory variable
    __private float sum = 0;
    
    for (int i = 0; i < numberOfVariables; i++) {
        sum = 0;
        for (int j = 0; j < numberOfVariables; j++) {
            sum += B2Array[(numberOfVariables * i) + j] * (diagD[j] * randomArray[j]);
        }
        //write to global memory
        arx2Array[(numberOfVariables * globalId) + i] = xmean[i] + sigma * sum;
    }
}