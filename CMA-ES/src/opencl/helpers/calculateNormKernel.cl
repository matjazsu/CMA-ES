#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void calculateNorm(int numberOfVariables, //read
                            __global float* ps, //read
                            __local float* psLocal, //set
                            __global float* psxps //set
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
    
    //Calculate norm(ps)^2
    
    //copy to local memory
    psLocal[localId] = (globalId < numberOfVariables) ? ps[globalId] : 0;
    
    //barrier call
    barrier(CLK_LOCAL_MEM_FENCE);
    
    psLocal[localId] = psLocal[localId] * psLocal[localId];
    
    for(int offset = wgSize; offset > 0; offset >>= 1)
    {
        if(localId < offset && localId + offset < wgSize)
        {
            psLocal[localId] += psLocal[localId + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    //barrier call
    barrier(CLK_LOCAL_MEM_FENCE);
    
    //only the first work item writes to global memory
    if(localId == 0)
    {
        psxps[groupId] = psLocal[0];
    }
    
}