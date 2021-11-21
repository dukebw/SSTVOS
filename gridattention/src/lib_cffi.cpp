// All functions assume that input and output tensors are already initialized
// and have the correct dimensions
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include "gridattn.h"

int
gridattn_map_forward_cuda(const at::Tensor& attn,
                          const at::Tensor& val,
                          at::Tensor& out)
{
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int N = val.size(0);
        int C = val.size(1);
        int T = val.size(2);
        int H = val.size(3);
        int W = val.size(4);

        const float *attn_data = attn.data_ptr<float>();
        const float *val_data = val.data_ptr<float>();
        float *out_data = out.data_ptr<float>();

        return _gridattn_map_forward_cuda(N, C, T, H, W, attn_data, val_data, out_data, stream);
}

int
gridattn_map_backward_cuda(const at::Tensor& dout,
                           const at::Tensor& attn,
                           const at::Tensor& val,
                           at::Tensor& dattn,
                           at::Tensor& dval)
{
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

        int N = dout.size(0);
        int C = dout.size(1);
        int T = dout.size(2);
        int H = dout.size(3);
        int W = dout.size(4);

        const float *dout_data = dout.data_ptr<float>();
        const float *attn_data = attn.data_ptr<float>();
        const float *val_data = val.data_ptr<float>();
        float *dattn_data = dattn.data_ptr<float>();
        float *dval_data = dval.data_ptr<float>();

        return _gridattn_map_backward_cuda(N,
                                           C,
                                           T,
                                           H,
                                           W,
                                           dout_data,
                                           attn_data,
                                           val_data,
                                           dattn_data,
                                           dval_data,
                                           stream);
}

int
gridattn_forward_cuda(const at::Tensor& query,
                      const at::Tensor& key,
                      at::Tensor& attnscores)
{
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int N = query.size(0);
        int C = query.size(1);
        int T = query.size(2);
        int H = query.size(3);
        int W = query.size(4);

        return _gridattn_forward_cuda(N,
                                      C,
                                      T,
                                      H,
                                      W,
                                      query.data_ptr<float>(),
                                      key.data_ptr<float>(),
                                      attnscores.data_ptr<float>(),
                                      stream);
}

int
gridattn_backward_cuda(const at::Tensor& dattn,
                       const at::Tensor& query,
                       const at::Tensor& key,
                       at::Tensor& dquery,
                       at::Tensor& dkey)
{
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
        int N = query.size(0);
        int C = query.size(1);
        int T = query.size(2);
        int H = query.size(3);
        int W = query.size(4);

        return _gridattn_backward_cuda(N,
                                       C,
                                       T,
                                       H,
                                       W,
                                       dattn.data_ptr<float>(),
                                       query.data_ptr<float>(),
                                       key.data_ptr<float>(),
                                       dquery.data_ptr<float>(),
                                       dkey.data_ptr<float>(),
                                       stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gridattn_forward_cuda",
          &gridattn_forward_cuda,
          "Grid attention forward CUDA");
    m.def("gridattn_backward_cuda",
          &gridattn_backward_cuda,
          "Grid attention backward CUDA");
    m.def("gridattn_map_forward_cuda",
          &gridattn_map_forward_cuda,
          "Grid attention map forward CUDA");
    m.def("gridattn_map_backward_cuda",
          &gridattn_map_backward_cuda,
          "Grid attention map backward CUDA");
}
