#define SIGMOID(inp) (1.0f / (1 + exp(-inp)))
#define RELU(inp) (inp > 0 ? inp : 0)
#define global_idx(x_idx, y_idx, m) (x_idx * m + y_idx)

__kernel void add_bias_helper(__global float *inp,
                              __global float *bias,
                              __global float *out,
                              ushort m,
                              int x_idx,
                              int y_idx)
{
    int x = global_idx(x_idx, y_idx, m);
    out[x] = inp[x] + bias[y_idx];
}

__kernel void add_bias(__global float *inp,
                       __global float *bias,
                       __global float *out,
                       ushort m)
{
    // global space:  inp.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    add_bias_helper(inp, bias, out, m, x_idx, y_idx);
}
/*
__kernel void matMul_helper(__global float *a,
                            __global float *b,
                            __global float *c,
                            ushort n,
                            ushort m,
                            ushort p,
                            int x_idx)
{
    c[x_idx] = 0.0f;
    int rowC = x_idx/p;
    int colC = x_idx%p;
    __global float *pA = &a[rowC*m];
    __global float *pB = &b[colC];
    //#pragma unroll
    for(int k=0; k<m; k++)
    {
        pB = &b[colC+k*p];
        c[x_idx] += (*(pA++))*(*pB);
    }
}

__kernel void matMul(__global float *a,
                     __global float *b,
                     __global float *c,
                     ushort n,
                     ushort m,
                     ushort p)
{
    // global space:  len(c) = n*p
    int x_idx = get_global_id(0);
    matMul_helper(a, b, c, n, m, p, x_idx);
}
*/
__kernel void matMul(__global const float* a,
                     __global const float* b,
                     __global float* result,
                     const ushort M,
                     const ushort N,
                     const ushort P)
{
    int idx = get_global_id(0);
    int k = 0;
    float temp = 0.0f;
    int i = idx / P;
    int j = idx % P;
    for( k = 0; k < N; k++)
        temp += a[ i*N + k ] * b[ k*P + j ];
    result[idx] = temp;

}


__kernel void transpose_helper(__global float *a_t,
                        __global float *a,
                        ushort n,
                        ushort m,
                        int x_idx,
                        int y_idx)
{
    a_t[global_idx(y_idx, x_idx, n)] = a[global_idx(x_idx, y_idx, m)];
}

__kernel void transpose(__global float *a_t,
                        __global float *a,
                        ushort n,
                        ushort m)
{
    // global space:  a.shape
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    transpose_helper(a_t, a, n, m, x_idx, y_idx);
}

__kernel void relu(__global float *inp,
                   __global float *out,
                   ushort m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    out[x] = RELU(inp[x]);
}

__kernel void sigmoid(__global float *inp,
                      __global float *out,
                      ushort m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    out[x] = SIGMOID(inp[x]);
}

__kernel void interaction_cat(__global float *sender,
                              __global float *receiver,
                              __global float *ri,
                              __global float *out,
                              ushort m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    if(x_idx < 3){
        out[x] = sender[x];
    } else if(x_idx < 6){
        out[x] = receiver[x - (3 * m)];
    } else {
        out[x] = ri[x - (6 * m)];
    }
}

__kernel void aggregate_cat(__global float *obj_t,
                              __global float *effect_receiver,
                              __global float *out,
                              ushort m)
{
    int x_idx = get_global_id(0);
    int y_idx = get_global_id(1);
    int x = global_idx(x_idx, y_idx, m);
    if(x_idx < 3){
        out[x] = obj_t[x];
    } else {
        out[x] = effect_receiver[x - (3 * m)];
    }
}