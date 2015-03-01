#if defined(cl_khr_fp64)  // Khronos extension
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

/* Number of kernels = populationSize */
__kernel void updateDistributionHelper(int N,
                                       __global float* artmp22array,
                                       __global float* B2Array,
                                       __global float* diagD,
                                       __global float* invsqrtC2array,
                                       __global float* C2Array,
                                       __global float* offdiag
                                       )
{
    //Get properties of current work item
    int globalId = get_global_id(0);
    
    if(globalId > N){
        return;
    }
    
    for (int j = 0; j <= globalId; j++) {
        B2Array[(N * globalId) + j] = B2Array[(N * j) + globalId] = C2Array[(N * globalId) + j];
    }
    
    diagD[globalId] = sqrt(diagD[globalId]);
    
    for (int j = 0; j < N; j++) {
        artmp22array[(N * globalId) + j] = B2Array[(N * globalId) + j] * (1 / diagD[j]);
    }
    
    for (int j = 0; j < N; j++) {
        invsqrtC2array[(N * globalId) + j] = 0;
        for (int k = 0; k < N; k++) {
            invsqrtC2array[(N * globalId) + j] += artmp22array[(N * globalId) + k] * B2Array[(N * j) + k];
        }
    }
    
    //tred2
    
    //  This is derived from the Algol procedures tred2 by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.
    
    diagD[globalId] = B2Array[(N * (N-1)) + globalId];
    
    //TO-DO: parallel execution
    if(globalId == 0){
        
        for(int i = N-1; i > 0; i--){
            __private float scale = 0;
            __private float h = 0;
            for (int k = 0; k < i; k++) {
                scale = scale + fabs(diagD[k]);
            }
            if (scale == 0) {
                offdiag[i] = diagD[i-1];
                for (int j = 0; j < i; j++) {
                    diagD[j] = B2Array[(N * (i-1)) + j];
                    B2Array[(N * i) + j] = 0;
                    B2Array[(N * j) + i] = 0;
                }
            }
            else{
                // Generate Householder vector
                
                for (int k = 0; k < i; k++) {
                    diagD[k] /= scale;
                    h += diagD[k] * diagD[k];
                }
                __private float f = diagD[i-1];
                __private float g = sqrt(h);
                if (f > 0) {
                    g = -g;
                }
                offdiag[i] = scale * g;
                h = h - f * g;
                diagD[i-1] = f - g;
                for (int j = 0; j < i; j++) {
                    offdiag[j] = 0;
                }
                
                // Apply similarity transformation to remaining columns
                
                for (int j = 0; j < i; j++) {
                    f = diagD[j];
                    B2Array[(N * j) + i] = f;
                    g = offdiag[j] + B2Array[(N * j) + j] * f;
                    for (int k = j+1; k <= i-1; k++) {
                        g += B2Array[(N * k) + j] * diagD[k];
                        offdiag[k] += B2Array[(N * k) + j] * f;
                    }
                    offdiag[j] = g;
                }
                f = 0;
                for (int j = 0; j < i; j++) {
                    offdiag[j] /= h;
                    f += offdiag[j] * diagD[j];
                }
                __private float hh = f / (h + h);
                for (int j = 0; j < i; j++) {
                    offdiag[j] -= hh * diagD[j];
                }
                for (int j = 0; j < i; j++) {
                    f = diagD[j];
                    g = offdiag[j];
                    for (int k = j; k <= i-1; k++) {
                        B2Array[(N * k) + j] -= (f * offdiag[k] + g * diagD[k]);
                    }
                    diagD[j] = B2Array[(N * (i-1)) + j];
                    B2Array[(N * i) + j] = 0;
                }
            }
            diagD[i] = h;
        }
        
        // Accumulate transformations
        for (int i = 0; i < N-1; i++) {
            B2Array[(N * (N-1)) + i] = B2Array[(N * i) + i];
            B2Array[(N * i) + i] = 1;
            __private float h = diagD[i+1];
            if (h != 0) {
                for (int k = 0; k <= i; k++) {
                    diagD[k] = B2Array[(N * k) + i+1] / h;
                }
                for (int j = 0; j <= i; j++) {
                    __private float g = 0;
                    for (int k = 0; k <= i; k++) {
                        g += B2Array[(N * k) + (i+1)] * B2Array[(N * k) + j];
                    }
                    for (int k = 0; k <= i; k++) {
                        B2Array[(N * k) + j] -= g * diagD[k];
                    }
                }
            }
            for (int k = 0; k <= i; k++) {
                B2Array[(N * k) + (i+1)] = 0;
            }
        }
        for (int j = 0; j < N; j++) {
            diagD[j] = B2Array[(N * (N-1)) + j];
            B2Array[(N * (N-1)) + j] = 0;
        }
        B2Array[(N * (N-1)) + (N-1)] = 1;
        offdiag[0] = 0;
    }
    
    //barrier call
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    //tql2
    
    //  This is derived from the Algol procedures tql2, by
    //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
    //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
    //  Fortran subroutine in EISPACK.
    
    if(globalId == N - 1){
        offdiag[globalId] = 0;
    }
    else{
        offdiag[globalId] = offdiag[globalId + 1];
    }
    
    //TO-DO: parallel execution
    if(globalId == 0){
        
        __private float f = 0;
        __private float tst1 = 0;
        
        __private float x = 2;
        __private float y = -52;
        __private float eps = pow(x,y);
        
        for(int l = 0; l < N; l++){
            
            tst1 = fmax(tst1, fabs(diagD[l] + fabs(offdiag[l])));
            
            __private int m = l;
            while(m < N){
                if(fabs(offdiag[l]) <= eps*tst1){
                    break;
                }
                m++;
            }
            
            if(m > l){
                while(fabs(offdiag[l]) > eps*tst1){
                    
                    // Compute implicit shift
                    __private float g = diagD[l];
                    __private float p = ((diagD[l + 1] - g) / (2 * offdiag[l]));
                    __private float r = hypot(p,1);
                    
                    if(p < 0){
                        r = -r;
                    }
                    
                    diagD[l] = offdiag[l] / (p + r);
                    diagD[l + 1] = offdiag[l] * (p + r);
                    __private float dl1 = diagD[l + 1];
                    __private float h = g - diagD[l];
                    for (int i = l+2; i < N; i++) {
                        diagD[i] -= h;
                    }
                    f = f + h;
                    
                    // Implicit QL transformation
                    
                    p = diagD[m];
                    __private float c = 1;
                    __private float c2 = c;
                    __private float c3 = c;
                    __private float el1 = offdiag[l+1];
                    __private float s = 0;
                    __private float s2 = 0;
                    for (int i = m-1; i >= l; i--) {
                        c3 = c2;
                        c2 = c;
                        s2 = s;
                        g = c * offdiag[i];
                        h = c * p;
                        r = hypot(p,offdiag[i]);
                        offdiag[i+1] = s * r;
                        s = offdiag[i] / r;
                        c = p / r;
                        p = c * diagD[i] - s * g;
                        diagD[i+1] = h + s * (c * g + s * diagD[i]);
                        
                        // Accumulate transformation.
                        
                        for (int k = 0; k < N; k++) {
                            h = B2Array[(N * k) + (i + 1)];
                            B2Array[(N * k) + (i + 1)] = s * B2Array[(N * k) + i] + c * h;
                            B2Array[(N * k) + i] = c * B2Array[(N * k) + i] - s * h;
                        }
                    }
                    p = -s * s2 * c3 * el1 * offdiag[l] / dl1;
                    offdiag[l] = s * p;
                    diagD[l] = c * p;
                }
            }
            
            diagD[l] = diagD[l] + f;
            offdiag[l] = 0;
        }
        
        // Sort eigenvalues and corresponding vectors.
        
        for (int i = 0; i < N-1; i++) {
            __private int k = i;
            __private float p = diagD[i];
            for (int j = i+1; j < N; j++) {
                if (diagD[j] < p) { // NH find smallest k>i
                    k = j;
                    p = diagD[j];
                }
            }
            if (k != i) {
                diagD[k] = diagD[i]; // swap k and i
                diagD[i] = p;
                for (int j = 0; j < N; j++) {
                    p = B2Array[(N * j) + i];
                    B2Array[(N * j) + i] = B2Array[(N * j) + k];
                    B2Array[(N * j) + k] = p;
                }
            }
        }
    }
}