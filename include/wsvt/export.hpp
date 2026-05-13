#pragma once

#if defined(WSVT_SHARED_BUILD)
    #if defined(_MSC_VER)
        #if defined(wsvt_core_EXPORTS)
            #define WSVT_API __declspec(dllexport)
        #else
            #define WSVT_API __declspec(dllimport)
        #endif
    #else
        #define WSVT_API __attribute__((visibility("default")))
    #endif
#else
    #define WSVT_API
#endif
