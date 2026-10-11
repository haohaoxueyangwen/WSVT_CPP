#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// Helper to find max location and value
void find_max(float* corr, int h, int w, int* max_y, int* max_x, float* max_val) {
    *max_val = -1e30f; // Very small number
    *max_y = 0;
    *max_x = 0;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            float val = corr[i * w + j];
            if (val > *max_val) {
                *max_val = val;
                *max_y = i;
                *max_x = j;
            }
        }
    }
}

// Sub-pixel refinement
void sub_pixel_refinement(float* corr, int h, int w, int max_y, int max_x, float* dy, float* dx) {
    // 3x3 neighborhood
    // c00 c01 c02
    // c10 c11 c12
    // c20 c21 c22
    // center is c11 at (max_y, max_x)
    
    // Match numpy.pad(..., mode='edge') in the Python reference.  Returning
    // zero refinement at a search boundary is a different estimator.
    int ym = max_y > 0 ? max_y - 1 : max_y;
    int yp = max_y < h - 1 ? max_y + 1 : max_y;
    int xm = max_x > 0 ? max_x - 1 : max_x;
    int xp = max_x < w - 1 ? max_x + 1 : max_x;
    float c00 = corr[ym * w + xm];
    float c01 = corr[ym * w + max_x];
    float c02 = corr[ym * w + xp];
    float c10 = corr[max_y * w + xm];
    float c11 = corr[max_y * w + max_x];
    float c12 = corr[max_y * w + xp];
    float c20 = corr[yp * w + xm];
    float c21 = corr[yp * w + max_x];
    float c22 = corr[yp * w + xp];

    float grad_y = (c21 - c01) / 2.0f;
    float grad_yy = c21 + c01 - 2.0f * c11;
    float grad_x = (c12 - c10) / 2.0f;
    float grad_xx = c12 + c10 - 2.0f * c11;
    float grad_xy = (c22 - c20 - c02 + c00) / 4.0f;

    float det = grad_xx * grad_yy - grad_xy * grad_xy;
    if (det != 0.0f) {
        *dx = -(grad_yy * grad_x - grad_xy * grad_y) / det;
        *dy = -(grad_xx * grad_y - grad_xy * grad_x) / det;
    } else {
        *dx = 0;
        *dy = 0;
    }
}

EXPORT void calculate_displacement_block(
    float* img_wa_stack,    // [dim_y, dim_x, depth]
    float* ref_wa_stack,    // [ref_h, ref_w, depth]
    float* disp_guess_y,    // [dim_y, dim_x]
    float* disp_guess_x,    // [dim_y, dim_x]
    int dim_y,
    int dim_x,
    int depth,
    int ref_h,
    int ref_w,
    int n_pad,
    int cal_half_window,
    float* out_disp_y,      // [dim_y, dim_x]
    float* out_disp_x       // [dim_y, dim_x]
) {
    int window_size = 2 * cal_half_window + 1;
    float* corr_map = (float*)malloc(window_size * window_size * sizeof(float));

    for (int y = 0; y < dim_y; y++) {
        for (int x = 0; x < dim_x; x++) {
            // Current pixel in img_wa_stack
            float* img_vec = &img_wa_stack[(y * dim_x + x) * depth];

            // Reference block start
            int guess_y = (int)disp_guess_y[y * dim_x + x];
            int guess_x = (int)disp_guess_x[y * dim_x + x];
            
            int ref_start_y = n_pad + y + guess_y;
            int ref_start_x = n_pad + x + guess_x;

            // Calculate correlation map
            for (int wy = 0; wy < window_size; wy++) {
                for (int wx = 0; wx < window_size; wx++) {
                    int ry = ref_start_y + wy;
                    int rx = ref_start_x + wx;
                    
                    float sum_sq_diff = 0.0f;
                    if (ry >= 0 && ry < ref_h && rx >= 0 && rx < ref_w) {
                        float* ref_vec = &ref_wa_stack[(ry * ref_w + rx) * depth];
                        for (int k = 0; k < depth; k++) {
                            float diff = img_vec[k] - ref_vec[k];
                            sum_sq_diff += diff * diff;
                        }
                    } else {
                        sum_sq_diff = 1e30f; // Large distance
                    }
                    corr_map[wy * window_size + wx] = -sum_sq_diff;
                }
            }

            // Find max
            int max_y, max_x;
            float max_val;
            find_max(corr_map, window_size, window_size, &max_y, &max_x, &max_val);

            // Sub-pixel
            float sub_dy, sub_dx;
            sub_pixel_refinement(corr_map, window_size, window_size, max_y, max_x, &sub_dy, &sub_dx);

            // Result
            // YY_axis[max_y, max_x] = max_y - cal_half_window
            float final_dy = sub_dy + (max_y - cal_half_window);
            float final_dx = sub_dx + (max_x - cal_half_window);
            if (final_dy > cal_half_window) final_dy = (float)cal_half_window;
            if (final_dy < -cal_half_window) final_dy = (float)-cal_half_window;
            if (final_dx > cal_half_window) final_dx = (float)cal_half_window;
            if (final_dx < -cal_half_window) final_dx = (float)-cal_half_window;

            out_disp_y[y * dim_x + x] = final_dy;
            out_disp_x[y * dim_x + x] = final_dx;
        }
    }
    free(corr_map);
}

// db2 coefficients
static const float LO_D[] = {-0.12940952255126037f, 0.2241438680420134f, 0.8365163037378079f, 0.48296291314453416f};
static const float HI_D[] = {-0.48296291314453416f, 0.8365163037378079f, -0.2241438680420134f, -0.12940952255126037f};

void dwt_step(float* input, int len, float* output_lo, float* output_hi, int out_len) {
    for (int i = 0; i < out_len; i++) {
        float lo = 0.0f;
        float hi = 0.0f;
        for (int k = 0; k < 4; k++) {
            int idx = 2 * i + 1 - k;
            float val = (idx >= 0 && idx < len) ? input[idx] : 0.0f;
            lo += val * LO_D[k];
            hi += val * HI_D[k];
        }
        if (output_lo) output_lo[i] = lo;
        if (output_hi) output_hi[i] = hi;
    }
}

EXPORT void wavelet_transform_db2(
    float* data,
    int n_pixels,
    int channels,
    int levels,
    int return_levels,
    int* coeff_offsets, // Offset in output for each coeff index [0..levels]
    int total_output_channels,
    float* output
) {
    float* buf1 = (float*)malloc(channels * sizeof(float));
    float* buf2 = (float*)malloc(channels * sizeof(float));
    float* detail = (float*)malloc(channels * sizeof(float));

    for (int p = 0; p < n_pixels; p++) {
        float* src_pixel = data + p * channels;
        float* dst_pixel = output + p * total_output_channels;
        
        // Copy input to buf1
        for(int i=0; i<channels; i++) buf1[i] = src_pixel[i];
        
        int current_len = channels;
        float* current_approx = buf1;
        float* next_approx = buf2;
        
        for (int k = 1; k <= levels; k++) {
            // Level k decomposition
            // cA_k (next_approx), cD_k (detail)
            // cD_k corresponds to coeffs index: levels - k + 1
            
            int cD_index = levels - k + 1;
            
            int next_len = (current_len + 4 - 1) / 2; 
            
            // We need to store detail if cD_index < return_levels
            float* d_ptr = (cD_index < return_levels) ? detail : NULL;
            
            dwt_step(current_approx, current_len, next_approx, d_ptr, next_len);
            
            if (cD_index < return_levels) {
                // Copy detail to output
                int offset = coeff_offsets[cD_index];
                for(int i=0; i<next_len; i++) {
                    dst_pixel[offset + i] = detail[i];
                }
            }
            
            // Swap buffers
            float* tmp = current_approx;
            current_approx = next_approx;
            next_approx = tmp;
            current_len = next_len;
        }
        
        // Store final approx cA_n (index 0)
        if (0 < return_levels) {
            int offset = coeff_offsets[0];
            for(int i=0; i<current_len; i++) {
                dst_pixel[offset + i] = current_approx[i];
            }
        }
    }
    
    free(buf1);
    free(buf2);
    free(detail);
}
