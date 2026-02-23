%%writefile trisword_dh.cu
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

// Tri-Sword Engine: Hardware Handle for DH Truncation Audit
struct TriSwordHandle {
    int8_t* ptr_rom;
    int32_t* ptr_all_positions;
    int32_t* ptr_rand_states;
    int int_max_batch;
    int int_total_positions;
};

__constant__ int32_t ARR_POSITION_OFFSETS[256];
__constant__ int32_t ARR_POSITION_LENGTHS[256];

// Generator: Xorshift32 interacting with truncated constants
__device__ inline int32_t fast_rand(int32_t& int_state) {
    int_state ^= int_state << 13;
    int_state ^= int_state >> 17; 
    int_state ^= int_state << 5;
    return int_state & 0x7FFFFFFF;
}

// DH Encode: Mapping PI-based generator to ROM Vertices
__global__ void trisword_encode_kernel(
    int8_t* ptr_messages,
    int32_t* ptr_encoded,
    int32_t* ptr_rand_states,
    int32_t* ptr_lookup_table,
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
            // Truncation Bias: The modulo logic that creates the "Wobble"
            ptr_encoded[sz_base_idx + i] = ptr_lookup_table[int_off + (int_val % int_len)];
        }
    }
    ptr_rand_states[sz_warp_id] = int_rand_state;
}

__global__ void trisword_decode_kernel(
    int32_t* ptr_encoded,
    int8_t* ptr_decoded,
    int8_t* ptr_rom,
    int int_batch_size,
    int int_msg_len
) {
    size_t sz_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t sz_total = (size_t)int_batch_size * int_msg_len;
    
    for (size_t i = sz_idx; i < sz_total; i += (size_t)gridDim.x * blockDim.x) {
        ptr_decoded[i] = ptr_rom[ptr_encoded[i] & 0xFFFF];
    }
}

intptr_t sharpen(int int_max_batch, torch::Tensor obj_rom_tensor, int int_rom_size) {
    TriSwordHandle* obj_h = new TriSwordHandle;
    obj_h->int_max_batch = int_max_batch;
    int8_t* ptr_rom_cpu = obj_rom_tensor.data_ptr<int8_t>();

    cudaMalloc(&obj_h->ptr_rom, int_rom_size);
    cudaMemcpy(obj_h->ptr_rom, ptr_rom_cpu, int_rom_size, cudaMemcpyHostToDevice);

    int32_t arr_offsets[256] = {0};
    int32_t arr_lengths[256] = {0};
    for (int i = 0; i < int_rom_size; i++) { arr_lengths[ptr_rom_cpu[i] & 0xFF]++; }

    int int_total_accum = 0;
    for (int i = 0; i < 256; i++) {
        arr_offsets[i] = int_total_accum;
        int_total_accum += arr_lengths[i];
    }
    obj_h->int_total_positions = int_total_accum;

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
    TriSwordHandle* obj_h = reinterpret_cast<TriSwordHandle*>(int_handle);
    if (obj_h) {
        cudaFree(obj_h->ptr_rom);
        cudaFree(obj_h->ptr_all_positions);
        cudaFree(obj_h->ptr_rand_states);
        delete obj_h;
    }
}

torch::Tensor ts_encode(intptr_t int_handle, torch::Tensor obj_msg) {
    TriSwordHandle* obj_h = reinterpret_cast<TriSwordHandle*>(int_handle);
    int int_batch = obj_msg.size(0);
    int int_len = obj_msg.size(1);
    auto obj_enc = torch::zeros({int_batch, int_len}, torch::dtype(torch::kInt32).device(obj_msg.device()));

    int int_threads = 256;
    int int_blocks = (int)((int_batch * 32LL + int_threads - 1) / int_threads);

    trisword_encode_kernel<<<int_blocks, int_threads>>>(
        obj_msg.data_ptr<int8_t>(),
        obj_enc.data_ptr<int32_t>(),
        obj_h->ptr_rand_states,
        obj_h->ptr_all_positions,
        int_batch,
        int_len
    );
    return obj_enc;
}

torch::Tensor ts_decode(intptr_t int_handle, torch::Tensor obj_enc) {
    TriSwordHandle* obj_h = reinterpret_cast<TriSwordHandle*>(int_handle);
    int int_batch = obj_enc.size(0);
    int int_len = obj_enc.size(1);
    auto obj_dec = torch::zeros({int_batch, int_len}, torch::dtype(torch::kInt8).device(obj_enc.device()));

    size_t sz_total = (size_t)int_batch * (size_t)int_len;
    int int_threads = 256;
    int int_blocks = (int)((sz_total + int_threads - 1) / int_threads);

    trisword_decode_kernel<<<int_blocks, int_threads>>>(
        obj_enc.data_ptr<int32_t>(),
        obj_dec.data_ptr<int8_t>(),
        obj_h->ptr_rom,
        int_batch,
        int_len
    );
    return obj_dec;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sharpen", &sharpen);
    m.def("sheath", &sheath);
    m.def("ts_encode", &ts_encode);
    m.def("ts_decode", &ts_decode);
}