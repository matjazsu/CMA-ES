#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

__kernel void updateDistribution(int N,
                                 int mu,
                                 float cs,
                                 int counteval,
                                 int lambda,
                                 float sigma,
                                 float chiN,
                                 float cc,
                                 float mueff,
                                 float cmu,
                                 float c1,
                                 __global float* ps,
                                 __local float* psLocal,
                                 __global float* psxps,
                                 __global float* arx2array,
                                 __global int* arindex, //read
                                 __global float* xmean,
                                 __global float* xold,
                                 __global float* pc,
                                 __global float* artmp,
                                 __global float* invsqrtC2array,
                                 __global float* C2array,
                                 __global float* weights //read
                                 )
{
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    int groupId = get_group_id(0);
    int wgSize = get_local_size(0);
    
    if(globalId >= N){
        return;
    }
    
    //############### calculate xmean and BDz~N(0,C) ###############//
    
    xold[globalId] = xmean[globalId];
    xmean[globalId] = 0;
    for (int iNk = 0; iNk < mu; iNk++) {
        //xmean[i] += weights[iNk] * arx[arindex[iNk]][i];
        xmean[globalId] += weights[iNk] * arx2array[(N * arindex[iNk]) + globalId];
    }
    
    //############### Cumulation: Update evolution paths ###############//
    
    artmp[globalId] = 0;
    //double value = (xmean[i] - xold[i]) / sigma;
    for (int j = 0; j < N; j++) {
        //artmp[globalId] += invsqrtC[i][j] * (xmean[j] - xold[j]) / sigma;
        artmp[globalId] += invsqrtC2array[(N * globalId) + j] * (xmean[j] - xold[j]) / sigma;
    }
    
    //############### cumulation for sigma (ps) ###############//
    
    ps[globalId] = (1 - cs) * ps[globalId] + sqrt(cs * (2 - cs) * mueff) * artmp[globalId];
    
    //############### calculate norm(ps)^2 ###############//
    //using OpenCL Work group REDUCTION
    
    //copy to local memory
    psLocal[localId] = (globalId < N) ? ps[globalId] : 0;
    
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
    
    //barrier call
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    //############### cumulation for covariance matrix (pc) ###############//
    
    //private variable
    __private int hsig = 0;
    
    if ((sqrt(psxps[0]) / sqrt(1 - pow(1 - cs, 2 * counteval/lambda)) / chiN) < (1,4 + 2 / (N + 1)))
    {
        hsig = 1;
    }
    
    pc[globalId] = (1 - cc) * pc[globalId] + hsig * sqrt(cc * (2 - cc) * mueff) * (xmean[globalId] - xold[globalId]) / sigma;
    
    //############### Adapt covariance matrix C ###############//
    
    for (int j = 0; j <= globalId; j++) {
        C2array[(N * globalId) + j] =  (1 - c1 - cmu)
        * C2array[(N * globalId) + j]
        + c1
        * (pc[globalId] * pc[j] + (1 - hsig) * cc * (2 - cc) * C2array[(N * globalId) + j]);
        for (int k = 0; k < mu; k++) {
            C2array[(N * globalId) + j] += cmu
            * weights[k]
            * (arx2array[(N * arindex[k]) + globalId] - xold[globalId])
            * (arx2array[(N * arindex[k]) + j] - xold[j]) / sigma
            / sigma;
        }
    }
    
}