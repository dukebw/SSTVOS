#ifndef __GRIDATTN__
#define __GRIDATTN__

#include "cuda_runtime.h"

/*
 * Exported functions
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
                           cudaStream_t stream);

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
                            cudaStream_t stream);


extern "C" int
_gridattn_forward_cuda(int N,
                       int C,
                       int T,
                       int H,
                       int W,
                       const float *query_data,
                       const float *key_data,
                       float *attnscores_data,
                       cudaStream_t stream);

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
                        cudaStream_t stream);

#endif /* __GRIDATTN__ */
