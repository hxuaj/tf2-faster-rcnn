// #include <iostream>

// #define CHECK(call) \
//     do \
//     { \
//         const cudaError_t error_code = call; \
//         if (error_code != cudaSuccess) \
//         { \
//             std::cerr << "CUDA Error:" << std::endl; \
//             std::cerr << "    File:       " << __FILE__ << std::endl; \
//             std::cerr << "    Line:       " << __LINE__ << std::endl; \
//             std::cerr << "    Error code: " << error_code << std::endl; \
//             std::cerr << "    Error text: " << cudaGetErrorString(error_code) << std::endl; \
//             exit(1); \
//         } \
//     } while (0)

// void _set_device(int device_id)
// {
//     int cur_device_id;
//     CHECK(cudaGetDevice(&cur_device_id));
//     if (cur_device_id == device_id)
//     {
//         return;
//     }
//     CHECK(cudaSetDevice(device_id));
// }

void nms(int *keep, const float *boxes, const int n, const int m,
         const int max_out, const float thresh, int *num_out, int device_id);