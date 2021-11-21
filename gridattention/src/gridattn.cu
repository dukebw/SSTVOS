#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

#include "gridattn.h"

__global__ void
gridattn_map_forward_kernel(const float *attn,
                            const float *val,
                            float *out,
                            int N,
                            int C,
                            int T,
                            int H,
                            int W)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int thread_yidx = blockIdx.y*blockDim.y + threadIdx.y;
        int t = thread_yidx % T;
        int y = thread_yidx / T;
        int c = blockIdx.z;
        int HW = H * W;
        int HWT = HW * T;
        int zdim = H + W + T - 2;
        int cHWT = c * HWT;
        int CHWT = C * HWT;

        if ((x >= W) || (y >= H) || (t >= T) || (c >= C))
                return;

        int xyt_offset = t*HW + y*W + x;
        for (int Nidx = 0;
             Nidx < N;
             ++Nidx) {
                float accum = 0.0f;

                int channel_offset = Nidx*CHWT + cHWT;
                int attn_Nxyt_offset = Nidx*zdim*HWT + xyt_offset;
                for (int z = 0;
                     z < W;
                     ++z) {
                        float attn_zxyt = attn[attn_Nxyt_offset + z*HWT];
                        float val_iyt = val[channel_offset + t*HW + y*W + z];

                        accum += attn_zxyt * val_iyt;
                }

                for (int z = W;
                     z < (W + H - 1);
                     ++z) {
                        int j = z - W;
                        j = (j < y) ? j : (j + 1);

                        float attn_zxyt = attn[attn_Nxyt_offset + z*HWT];
                        float val_xjt = val[channel_offset + t*HW + j*W + x];

                        accum += attn_zxyt * val_xjt;
                }

                for (int z = (W + H - 1);
                     z < (W + H + T - 2);
                     ++z) {
                        int k = z - (W + H - 1);
                        k = (k < t) ? k : (k + 1);

                        float attn_zxyt = attn[attn_Nxyt_offset + z*HWT];
                        float val_xyk = val[channel_offset + k*HW + y*W + x];

                        accum += attn_zxyt * val_xyk;
                }

                out[channel_offset + xyt_offset] = accum;
        }
}

__global__ void
gridattn_map_backward_kernel_dattn(const float *dout,
                                   const float *val,
                                   float *dattn,
                                   int N,
                                   int C,
                                   int T,
                                   int H,
                                   int W)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int thread_yidx = blockIdx.y*blockDim.y + threadIdx.y;
        int t = thread_yidx % T;
        int y = thread_yidx / T;
        int z = blockIdx.z;
        int HW = H * W;
        int HWT = HW * T;
        int zdim = H + W + T - 2;
        int zHWT = z * HWT;
        int zdimHWT = zdim * HWT;
        /* TODO(brendan): divide by sqrt(C), in backward as well. */

        if ((x >= W) || (y >= H) || (t >= T) || (z >= (H + W + T - 2)))
                return;

        int i;
        int j;
        int k;
        if (z < W) {
                i = z;
                j = y;
                k = t;
        } else if (z < (H + W - 1)) {
                i = x;
                j = z - W;
                j = j < y ? j : j + 1;
                k = t;
        } else {
                i = x;
                j = y;
                k = z - (H + W - 1);
                k = k < t ? k : k + 1;
        }
        int ijk_offset = k*HW + j*W + i;
        int xyt_offset = t*HW + y*W + x;
        int zxyt_offset = zHWT + xyt_offset;

        for (int Nidx = 0;
             Nidx < N;
             ++Nidx) {
                float accum = 0.0f;

                for (int Cidx = 0;
                     Cidx < C;
                     ++Cidx) {
                        int channel_offset = (Nidx*C + Cidx)*HWT;
                        float dout_xyt = dout[channel_offset + xyt_offset];
                        float val_ijk = val[channel_offset + ijk_offset];

                        accum += dout_xyt * val_ijk;
                }

                dattn[Nidx*zdimHWT + zxyt_offset] = accum;
        }
}

__global__ void
gridattn_map_backward_kernel_dval(const float *attn,
                                  const float *dout,
                                  float *dval,
                                  int N,
                                  int C,
                                  int T,
                                  int H,
                                  int W)
{
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int thread_yidx = blockIdx.y*blockDim.y + threadIdx.y;
        int k = thread_yidx % T;
        int j = thread_yidx / T;
        int c = blockIdx.z;
        int HW = H * W;
        int HWT = HW * T;
        int zdim = H + W + T - 2;
        int cHWT = c * HWT;
        int CHWT = C * HWT;

        if ((i >= W) || (j >= H) || (k >= T) || (c >= C))
                return;

        int ijk_offset = k*HW + j*W + i;
        for (int Nidx = 0;
             Nidx < N;
             ++Nidx) {
                float accum = 0.0f;

                int channel_offset = Nidx*CHWT + cHWT;
                int attn_N_offset = Nidx * zdim * HWT;
                for (int x = 0;
                     x < W;
                     ++x) {
                        int xjk_offset = k*HW + j*W + x;
                        float attn_zxjk = attn[attn_N_offset + i*HWT + xjk_offset];
                        float dout_xjk = dout[channel_offset + xjk_offset];

                        accum += attn_zxjk * dout_xjk;
                }

                for (int y = 0;
                     y < H;
                     ++y) {
                        if (y == j)
                                continue;

                        int z = (y > j) ? j : (j - 1);
                        z += W;

                        int iyk_offset = k*HW + y*W + i;
                        float attn_ziyk = attn[attn_N_offset + z*HWT + iyk_offset];
                        float dout_iyk = dout[channel_offset + iyk_offset];

                        accum += attn_ziyk * dout_iyk;
                }

                for (int t = 0;
                     t < T;
                     ++t) {
                        if (t == k)
                                continue;

                        int z = (t > k) ? k : (k - 1);
                        z += W + H - 1;

                        int ijt_offset = t*HW + j*W + i;
                        float attn_zxyt = attn[attn_N_offset + z*HWT + ijt_offset];
                        float dout_ijt = dout[channel_offset + ijt_offset];

                        accum += attn_zxyt * dout_ijt;
                }

                dval[channel_offset + ijk_offset] = accum;
        }
}

__global__ void
gridattn_forward_kernel(const float *query,
                        const float *key,
                        float *attnscores,
                        int N,
                        int C,
                        int T,
                        int H,
                        int W)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int thread_yidx = blockIdx.y*blockDim.y + threadIdx.y;
        int t = thread_yidx % T;
        int y = thread_yidx / T;
        int z = blockIdx.z;
        int HW = H * W;
        int HWT = HW * T;
        int zdim = H + W + T - 2;
        int zHWT = z * HWT;
        int zdimHWT = zdim * HWT;

        if ((x >= W) || (y >= H) || (t >= T) || (z >= (H + W + T - 2)))
                return;

        int i;
        int j;
        int k;
        if (z < W) {
                i = z;
                j = y;
                k = t;
        } else if (z < (H + W - 1)) {
                i = x;
                j = z - W;
                j = j < y ? j : j + 1;
                k = t;
        } else {
                i = x;
                j = y;
                k = z - (H + W - 1);
                k = k < t ? k : k + 1;
        }
        int ijk_offset = k*HW + j*W + i;
        int xyt_offset = t*HW + y*W + x;
        int zxyt_offset = zHWT + xyt_offset;

        for (int Nidx = 0;
             Nidx < N;
             ++Nidx) {
                float accum = 0.0f;
                for (int Cidx = 0;
                     Cidx < C;
                     ++Cidx) {
                        int channel_offset = (Nidx*C + Cidx)*HWT;
                        float query_xyt = query[channel_offset + xyt_offset];
                        float key_ijk = key[channel_offset + ijk_offset];

                        accum += query_xyt * key_ijk;
                }

                attnscores[Nidx*zdimHWT + zxyt_offset] = accum;
        }
}

__global__ void
gridattn_backward_kernel_dquery(const float *dattn,
                                const float *key,
                                float *dquery,
                                int N,
                                int C,
                                int T,
                                int H,
                                int W)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int thread_yidx = blockIdx.y*blockDim.y + threadIdx.y;
        int t = thread_yidx % T;
        int y = thread_yidx / T;
        int c = blockIdx.z;
        int HW = H * W;
        int HWT = HW * T;
        int zdim = H + W + T - 2;
        int cHWT = c * HWT;
        int CHWT = C * HWT;

        if ((x >= W) || (y >= H) || (t >= T) || (c >= C))
                return;

        int xyt_offset = t*HW + y*W + x;
        for (int Nidx = 0;
             Nidx < N;
             ++Nidx) {
                float accum = 0.0f;

                int channel_offset = Nidx*CHWT + cHWT;
                int attn_Nxyt_offset = Nidx*zdim*HWT + xyt_offset;
                for (int z = 0;
                     z < W;
                     ++z) {
                        float dattn_zxyt = dattn[attn_Nxyt_offset + z*HWT];
                        float key_iyt = key[channel_offset + t*HW + y*W + z];

                        accum += dattn_zxyt * key_iyt;
                }

                for (int z = W;
                     z < (W + H - 1);
                     ++z) {
                        int j = z - W;
                        j = (j < y) ? j : (j + 1);

                        float dattn_zxyt = dattn[attn_Nxyt_offset + z*HWT];
                        float key_xjt = key[channel_offset + t*HW + j*W + x];

                        accum += dattn_zxyt * key_xjt;
                }

                for (int z = (W + H - 1);
                     z < (W + H + T - 2);
                     ++z) {
                        int k = z - (W + H - 1);
                        k = (k < t) ? k : (k + 1);

                        float dattn_zxyt = dattn[attn_Nxyt_offset + z*HWT];
                        float key_xyk = key[channel_offset + k*HW + y*W + x];

                        accum += dattn_zxyt * key_xyk;
                }

                dquery[channel_offset + xyt_offset] = accum;
        }
}

__global__ void
gridattn_backward_kernel_dkey(const float *dattn,
                              const float *query,
                              float *dkey,
                              int N,
                              int C,
                              int T,
                              int H,
                              int W)
{
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int thread_yidx = blockIdx.y*blockDim.y + threadIdx.y;
        int k = thread_yidx % T;
        int j = thread_yidx / T;
        int c = blockIdx.z;
        int HW = H * W;
        int HWT = HW * T;
        int zdim = H + W + T - 2;
        int cHWT = c * HWT;
        int CHWT = C * HWT;

        if ((i >= W) || (j >= H) || (k >= T) || (c >= C))
                return;

        int ijk_offset = k*HW + j*W + i;
        for (int Nidx = 0;
             Nidx < N;
             ++Nidx) {
                float dkey_accum = 0.0f;

                int channel_offset = Nidx*CHWT + cHWT;
                int attn_N_offset = Nidx * zdim * HWT;
                for (int x = 0;
                     x < W;
                     ++x) {
                        int xjk_offset = k*HW + j*W + x;
                        float dattn_zxjk = dattn[attn_N_offset + i*HWT + xjk_offset];
                        float query_xjk = query[channel_offset + xjk_offset];

                        dkey_accum += dattn_zxjk * query_xjk;
                }

                for (int y = 0;
                     y < H;
                     ++y) {
                        if (y == j)
                                continue;

                        int z = (y > j) ? j : (j - 1);
                        z += W;

                        int iyk_offset = k*HW + y*W + i;
                        float dattn_ziyk = dattn[attn_N_offset + z*HWT + iyk_offset];
                        float query_iyk = query[channel_offset + iyk_offset];
                        dkey_accum += dattn_ziyk * query_iyk;
                }

                for (int t = 0;
                     t < T;
                     ++t) {
                        if (t == k)
                                continue;
#ifdef MASKED_GRIDATTENTION
                        if (k > t)
                                continue;
#endif /* MASKED_GRIDATTENTION */

                        int z = (t > k) ? k : (k - 1);
                        z += W + H - 1;

                        int ijt_offset = t*HW + j*W + i;
                        float dattn_zxyt = dattn[attn_N_offset + z*HWT + ijt_offset];
                        float query_ijt = query[channel_offset + ijt_offset];
                        dkey_accum += dattn_zxyt * query_ijt;
                }

                dkey[channel_offset + ijk_offset] = dkey_accum;
        }
}

/*
 * Implementations
 */
extern "C" int
_gridattn_map_forward_cuda(int N,
                           int C,
                           int T,
                           int H,
                           int W,
                           const float *attn,
                           const float *val,
                           float *out,
                           cudaStream_t stream)
{
        // Run kernel
        dim3 threads_per_block{32, 32};
        uint32_t d1 = (W + threads_per_block.x - 1)/threads_per_block.x;
        uint32_t d2 = (T*H + threads_per_block.y - 1)/threads_per_block.y;
        uint32_t d3 = C;
        dim3 blocks{d1, d2, d3};
        gridattn_map_forward_kernel<<<blocks, threads_per_block, 0, stream>>>(attn,
                                                                              val,
                                                                              out,
                                                                              N,
                                                                              C,
                                                                              T,
                                                                              H,
                                                                              W);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
                return 0;

        return 1;
}

extern "C" int
_gridattn_map_backward_cuda(int N,
                            int C,
                            int T,
                            int H,
                            int W,
                            const float *dout,
                            const float *attn,
                            const float *val,
                            float *dattn,
                            float *dval,
                            cudaStream_t stream)
{
        // Run kernel
        dim3 threads_per_block{32, 32};
        uint32_t d1 = (W + threads_per_block.x - 1) / threads_per_block.x;
        uint32_t d2 = (T*H + threads_per_block.y - 1) / threads_per_block.y;
        uint32_t d3 = H + W + T;
        dim3 blocks{d1, d2, d3};
        gridattn_map_backward_kernel_dattn<<<blocks, threads_per_block, 0, stream>>>(dout,
                                                                                     val,
                                                                                     dattn,
                                                                                     N,
                                                                                     C,
                                                                                     T,
                                                                                     H,
                                                                                     W);

        d3 = C;
        blocks = dim3{d1, d2, d3};
        gridattn_map_backward_kernel_dval<<<blocks, threads_per_block, 0, stream>>>(attn,
                                                                                    dout,
                                                                                    dval,
                                                                                    N,
                                                                                    C,
                                                                                    T,
                                                                                    H,
                                                                                    W);

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
                return 0;

        return 1;
}

extern "C" int
_gridattn_forward_cuda(int N,
                       int C,
                       int T,
                       int H,
                       int W,
                       const float *query_data,
                       const float *key_data,
                       float *attnscores_data,
                       cudaStream_t stream)
{
        dim3 threads_per_block{32, 32};
        uint32_t d1 = (W + threads_per_block.x - 1) / threads_per_block.x;
        uint32_t d2 = (T*H + threads_per_block.y - 1) / threads_per_block.y;
        uint32_t d3 = H + W + T;
        dim3 blocks{d1, d2, d3};
        gridattn_forward_kernel<<<blocks, threads_per_block, 0, stream>>>(query_data,
                                                                          key_data,
                                                                          attnscores_data,
                                                                          N,
                                                                          C,
                                                                          T,
                                                                          H,
                                                                          W);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
                return 0;

        return 1;
}

extern "C" int
_gridattn_backward_cuda(int N,
                        int C,
                        int T,
                        int H,
                        int W,
                        const float *dattn_data,
                        const float *query_data,
                        const float *key_data,
                        float *dquery_data,
                        float *dkey_data,
                        cudaStream_t stream)
{
        dim3 threads_per_block{32, 32};
        uint32_t d1 = (W + threads_per_block.x - 1)/threads_per_block.x;
        uint32_t d2 = (T*H + threads_per_block.y - 1)/threads_per_block.y;
        uint32_t d3 = C;
        dim3 blocks{d1, d2, d3};
        gridattn_backward_kernel_dquery<<<blocks, threads_per_block, 0, stream>>>(dattn_data,
                                                                                  key_data,
                                                                                  dquery_data,
                                                                                  N,
                                                                                  C,
                                                                                  T,
                                                                                  H,
                                                                                  W);

        gridattn_backward_kernel_dkey<<<blocks, threads_per_block, 0, stream>>>(dattn_data,
                                                                                query_data,
                                                                                dkey_data,
                                                                                N,
                                                                                C,
                                                                                T,
                                                                                H,
                                                                                W);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
                return 0;

        return 1;
}
