__kernel void mmul(const int N,   
    __global float* A, 
    __global float* B,  
    __global float* C,
    __local float* bwrk) {
    int i = get_global_id(0);  // this kernel runs per row, because managing many worker groups takes time. 
                               // Each row is a work item. And we have 64 work items per work group.
                               // This means that 1024/64=16 which means we have 16 work groups .
    int iloc = get_local_id(0);  // this is the worker item index in the worker group. each work group will move over the same columns.
    int nloc = get_local_size(0); // this will be 64.



    int j,k;
    float tmp = 0.0f;
    float awrk[1024]; // a work is in private memory as each work item is using a different row.
    for (k = 0; k < N; k++) {
        awrk[k] = A[i * N + k];
    }
    for (j = 0; j < N; j++) {
        for (k = iloc; k < N; k+=nloc) {
            bwrk[k] = B[k * N + j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        tmp = 0.0f;
        for (k = 0; k < N; k++) {
            tmp += awrk[k] * B[k];
        }
        C[i * N + j] = tmp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}