__kernel void mmul(const int N,   
    __global float* A, 
    __global float* B,  
    __global float* C) {
    int i = get_global_id(0);  // this kernel runs per row, because managing many worker groups takes time. 
                               // Each row is a work item. And we have 64 work items per work group.
                               // This means that 1024/64=16 which means we have 16 work groups .
    int j,k;
    float tmp = 0.0f;
    float awrk[1024];
    for (k = 0; k < N; k++) {
        awrk[k] = A[i * N + k];
    }
    for (j = 0; j < N; j++) {
        tmp = 0.0f;
        for (k = 0; k < N; k++) {
            tmp += awrk[k] * B[k * N + j];
        }
        C[i * N + j] = tmp;
    }
}