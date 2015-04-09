#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void storeBest(int populationSize, //read
                        int numberOfVariables, //read
                        __global float* arx2Array, //read
                        __local float* locArray, //set
                        __global float* sumEval, //set
                        __global float* bestSolutionEverArray, //set
                        __global float* arfitness //set
                        )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int wgSize = get_local_size(0);
    
    if(globalId > numberOfVariables){
        return;
    }
    
    for(int row = 0; row < populationSize; row++){
        
        locArray[localId] = 100 * (arx2Array[(numberOfVariables * row) + globalId+1] -  arx2Array[(numberOfVariables * row) + globalId] * arx2Array[(numberOfVariables * row) + globalId]) * (arx2Array[(numberOfVariables * row) + globalId+1] - arx2Array[(numberOfVariables * row) + globalId] * arx2Array[(numberOfVariables * row) + globalId]) + (arx2Array[(numberOfVariables * row) + globalId]-1) * (arx2Array[(numberOfVariables * row) + globalId]-1);
        
        for(int offset = wgSize; offset > 0; offset >>= 1)
        {
            if(localId < offset && localId + offset < wgSize)
            {
                locArray[localId] += locArray[localId + offset];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        //barrier call
        barrier(CLK_LOCAL_MEM_FENCE);
        
        //only the first work item writes to global memory
        if(localId == 0)
        {
            sumEval[groupId] = locArray[0];
            arfitness[row] = locArray[0];
        }
        
        //barrier call
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if(sumEval[groupId] < bestSolutionEverArray[numberOfVariables] || bestSolutionEverArray[numberOfVariables] == 0){
            //Set new best solution
            bestSolutionEverArray[globalId] = arx2Array[(numberOfVariables * 0) + globalId];
            if(globalId == 0){
                bestSolutionEverArray[numberOfVariables] = sumEval[groupId];
            }
        }
    }
}