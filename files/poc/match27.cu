%%writefile engine.cu
#include <torch/extension.h>
#include <cuda.h>

__global__ void tag_kernel(const uint8_t* src, uint8_t* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N - 1) {
        dst[i] = (src[i] == 1 && src[i + 1] == 1) ? 2 : src[i];
    }
}

torch::Tensor tag_pattern(torch::Tensor input) {
    auto output = torch::empty_like(input);
    tag_kernel<<<(input.numel() + 255) / 256, 256>>>(
        input.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), input.numel()
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tag", &tag_pattern, "Expert Write API");
}



%%writefile comparison.py
import torch
import time
from torch.utils.cpp_extension import load

# 1. THE EXPERT ENGINE (Your JIT Kernel)
expert = load(name="expert_engine", sources=["engine.cu"], verbose=False)

# 2. THE NORMAL METHOD (Generalist PyTorch logic)
def normal_tag(m):
    # This creates a shifted copy of the 10M elements (Memory Heavy!)
    mask = (m[:-1] == 1) & (m[1:] == 1)
    res = m.clone()
    res[:-1][mask] = 2
    return res

# DATA
data = torch.zeros(10_000_000, dtype=torch.uint8, device='cuda')
data[500:502] = 1

# TEST NORMAL
torch.cuda.synchronize()
t0 = time.time()
res_normal = normal_tag(data)
torch.cuda.synchronize()
t_normal = (time.time() - t0) * 1000

# TEST EXPERT
torch.cuda.synchronize()
t1 = time.time()
res_expert = expert.tag(data)
torch.cuda.synchronize()
t_expert = (time.time() - t1) * 1000

print(f"--- THE GAP ---")
print(f"Normal Method: {t_normal:.4f}ms")
print(f"Expert Engine: {t_expert:.4f}ms")
print(f"Speed Increase: {t_normal / t_expert:.2f}x faster")




--- THE GAP ---

Normal Method: 151.4311ms

Expert Engine: 0.5832ms

Speed Increase: 259.67x faster

