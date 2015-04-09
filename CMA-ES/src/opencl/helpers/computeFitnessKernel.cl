#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void computeFitness(int populationSize, //read
                             int numberOfVariables, //read
                             __global float* arfitness, //read
                             __global int* arindex //set
                             )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId > populationSize){
        return;
    }
    
    //Initial set arindex array
    arindex[globalId] = globalId;
    
    //Update arindex array
    if(globalId == 0){
        for (int i = 0; i < populationSize; i++) {
            for (int j = i + 1; j < populationSize; j++) {
                if (arfitness[i] > arfitness[j]) {
                    __private float temp = arfitness[i];
                    __private int tempIdx = arindex[i];
                    arfitness[i] = arfitness[j];
                    arfitness[j] = temp;
                    arindex[i] = arindex[j];
                    arindex[j] = tempIdx;
                } else if (arfitness[i] == arfitness[j]) {
                    if (arindex[i] > arindex[j]) {
                        __private float temp = arfitness[i];
                        __private int tempIdx = arindex[i];
                        arfitness[i] = arfitness[j];
                        arfitness[j] = temp;
                        arindex[i] = arindex[j];
                        arindex[j] = tempIdx;
                    }
                }
            }
        }
    }
}