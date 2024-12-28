#ifndef SAVE_BMP_INCLUDE
#define SAVE_BMP_INCLUDE

#include <stdint.h>

enum save_bmp_result {
    SAVE_BMP_SUCCESS,
    SAVE_BMP_SIZE_IS_ZERO,
    SAVE_BMP_SIZE_TOO_BIG,
    SAVE_BMP_CANT_OPEN_FILE,
    SAVE_BMP_WRITE_ERROR
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

    enum save_bmp_result save_bmp(const char* filename,
        uint32_t width, uint32_t height,
        const uint8_t* image);

    const char* save_bmp_str_result(enum save_bmp_result result);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // SAVE_BMP_INCLUDE