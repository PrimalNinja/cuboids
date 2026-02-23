%%writefile dh_bias.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

struct DHHandle {
    int8_t* ptr_rom;
    int32_t* ptr_all_positions;
    int32_t* ptr_rand_states;
    int int_max_batch;
};

__constant__ int32_t ARR_POSITION_OFFSETS[256];
__constant__ int32_t ARR_POSITION_LENGTHS[256];

__device__ inline int32_t fast_rand(int32_t& int_state) {
    int_state ^= int_state << 13;
    int_state ^= int_state >> 17; 
    int_state ^= int_state << 5;
    return int_state & 0x7FFFFFFF;
}

__global__ void bias_detector_kernel(
    int8_t* ptr_messages,
    int32_t* ptr_encoded,
    int32_t* ptr_rand_states,
    int32_t* ptr_lookup_table,
    int32_t* ptr_histogram, 
    int int_batch_size,
    int int_msg_len
) {
    size_t sz_warp_id = (size_t)((blockIdx.x * blockDim.x + threadIdx.x) / 32);
    int int_lane_id = threadIdx.x % 32;
    if (sz_warp_id >= (size_t)int_batch_size) return;
    
    int32_t int_rand_state = ptr_rand_states[sz_warp_id];
    int int_per_thread = (int_msg_len + 31) / 32;
    int int_start = int_lane_id * int_per_thread;
    int int_end = min(int_start + int_per_thread, int_msg_len);
    size_t sz_base_idx = sz_warp_id * (size_t)int_msg_len;
    
    for (int i = int_start; i < int_end; i++) {
        int int_byte = ptr_messages[sz_base_idx + i] & 0xFF;
        int int_off = ARR_POSITION_OFFSETS[int_byte];
        int int_len = ARR_POSITION_LENGTHS[int_byte];
        if (int_len > 0) {
            int32_t int_val = fast_rand(int_rand_state);
            // Hard boundary check to prevent SegFault
            int32_t selected_addr = ptr_lookup_table[int_off + (int_val % int_len)];
            atomicAdd(&ptr_histogram[selected_addr & 0xFFFF], 1);
            ptr_encoded[sz_base_idx + i] = selected_addr;
        }
    }
    ptr_rand_states[sz_warp_id] = int_rand_state;
}

intptr_t sharpen(int int_max_batch, torch::Tensor obj_rom_tensor, int int_rom_size) {
    DHHandle* obj_h = new DHHandle;
    obj_h->int_max_batch = int_max_batch;
    
    // Ensure ROM is on GPU
    torch::Tensor obj_gpu_rom = obj_rom_tensor.cuda().contiguous();
    int8_t* ptr_rom_cpu = obj_rom_tensor.cpu().data_ptr<int8_t>();
    
    cudaMalloc(&obj_h->ptr_rom, int_rom_size);
    cudaMemcpy(obj_h->ptr_rom, obj_gpu_rom.data_ptr<int8_t>(), int_rom_size, cudaMemcpyDeviceToDevice);
    
    int32_t arr_offsets[256] = {0}, arr_lengths[256] = {0};
    for (int i = 0; i < int_rom_size; i++) { arr_lengths[ptr_rom_cpu[i] & 0xFF]++; }
    int int_total_accum = 0;
    for (int i = 0; i < 256; i++) { arr_offsets[i] = int_total_accum; int_total_accum += arr_lengths[i]; }
    
    int32_t* arr_pos_cpu = new int32_t[int_total_accum];
    int32_t arr_curr[256] = {0};
    for (int i = 0; i < int_rom_size; i++) { 
        int b = ptr_rom_cpu[i] & 0xFF; 
        arr_pos_cpu[arr_offsets[b] + arr_curr[b]++] = i; 
    }
    
    cudaMemcpyToSymbol(ARR_POSITION_OFFSETS, arr_offsets, 256 * sizeof(int32_t));
    cudaMemcpyToSymbol(ARR_POSITION_LENGTHS, arr_lengths, 256 * sizeof(int32_t));
    cudaMalloc(&obj_h->ptr_all_positions, int_total_accum * sizeof(int32_t));
    cudaMemcpy(obj_h->ptr_all_positions, arr_pos_cpu, int_total_accum * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    cudaMalloc(&obj_h->ptr_rand_states, int_max_batch * sizeof(int32_t));
    int32_t* arr_seeds = new int32_t[int_max_batch];
    for (int i = 0; i < int_max_batch; i++) { arr_seeds[i] = i * 0x9e3779b1 + 12345; }
    cudaMemcpy(obj_h->ptr_rand_states, arr_seeds, int_max_batch * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    delete[] arr_pos_cpu; 
    delete[] arr_seeds;
    return reinterpret_cast<intptr_t>(obj_h);
}

void sheath(intptr_t int_handle) {
    DHHandle* obj_h = reinterpret_cast<DHHandle*>(int_handle);
    if (obj_h) { 
        cudaFree(obj_h->ptr_rom); 
        cudaFree(obj_h->ptr_all_positions); 
        cudaFree(obj_h->ptr_rand_states); 
        delete obj_h; 
    }
}

torch::Tensor slash_encode_bias(intptr_t int_handle, torch::Tensor obj_msg, torch::Tensor obj_hist) {
    DHHandle* obj_h = reinterpret_cast<DHHandle*>(int_handle);
    int int_batch = obj_msg.size(0), int_len = obj_msg.size(1);
    auto obj_enc = torch::zeros({int_batch, int_len}, torch::dtype(torch::kInt32).device(obj_msg.device()));
    int int_threads = 256;
    int int_blocks = (int)((int_batch * 32LL + int_threads - 1) / int_threads);
    
    bias_detector_kernel<<<int_blocks, int_threads>>>(
        obj_msg.data_ptr<int8_t>(), 
        obj_enc.data_ptr<int32_t>(), 
        obj_h->ptr_rand_states, 
        obj_h->ptr_all_positions, 
        obj_hist.data_ptr<int32_t>(), 
        int_batch, 
        int_len
    );
    return obj_enc;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sharpen", &sharpen);
    m.def("sheath", &sheath);
    m.def("slash_encode_bias", &slash_encode_bias);
}