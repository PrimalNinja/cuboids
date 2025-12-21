%%writefile engine.cu
#include <torch/extension.h>
#include <cuda.h>

// THE WRITE FUNCTION: Direct-to-metal structural tagging
__global__ void tag_kernel(const uint8_t* src, uint8_t* dst, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N - 1) {
        // Find pattern '1,1' and write '2' into the framebuffer
        dst[i] = (src[i] == 1 && src[i + 1] == 1) ? 2 : src[i];
    }
}

torch::Tensor tag_pattern(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int N = input.numel();
    tag_kernel<<<(N + 255) / 256, 256>>>(
        input.data_ptr<uint8_t>(), 
        output.data_ptr<uint8_t>(), 
        N
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tag", &tag_pattern, "Expert Write API");
}