#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    int height,width,mode,input_u16,low_memory,spatial_chunk,device,mapped_input;
    double pixel_m,distance_m,energy_ev;
} SpeckleConfig;
typedef struct {
    double reference_s,sample_solver_s,download_s,call_s,stages_s[8];
    uint64_t persistent_bytes,fft_workspace_bytes,device_allocation_bytes;
} SpeckleTiming;
// mode: 0 WXST / 1 WSVT. Seven output buffers are owned by the caller.
// WXST uses float32 HxW; WSVT uses uint16/float32 100xHxW. Fixed half10/L2/I1/cut1.
void* speckle_create(const SpeckleConfig* config);
void speckle_destroy(void* session);
const char* speckle_error(void);
int speckle_prepare(void* session,const void* reference,size_t bytes,double* seconds);
int speckle_solve(void* session,const void* sample,size_t bytes,void** outputs,SpeckleTiming* timing);
int speckle_copy_descriptor(void* session,int index,void* output,size_t bytes);
uint64_t speckle_descriptor_bytes(void* session,int index);
void* speckle_host_alloc(size_t bytes,int mapped);
int speckle_host_free(void* ptr);
int speckle_device_info(int device,char* text,size_t capacity);
#ifdef __cplusplus
}
#endif
