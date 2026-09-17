#pragma once

// Source literals and redirected output use UTF-8 on every platform.
// A real Windows console must decode those same bytes as UTF-8 too.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <cstdio>
#include <iostream>

namespace prp_console {
class utf8_output_guard {
    UINT original_code_page_ = 0;
    bool changed_ = false;

    static bool is_console(FILE* stream) noexcept {
        const int fd = _fileno(stream);
        if (fd < 0) return false;
        const intptr_t raw_handle = _get_osfhandle(fd);
        if (raw_handle == -1 || raw_handle == -2) return false;
        DWORD mode = 0;
        return GetConsoleMode(reinterpret_cast<HANDLE>(raw_handle), &mode) != 0;
    }

public:
    utf8_output_guard() noexcept {
        if (!is_console(stdout) && !is_console(stderr)) return;
        original_code_page_ = GetConsoleOutputCP();
        if (original_code_page_ != 0 && original_code_page_ != CP_UTF8)
            changed_ = SetConsoleOutputCP(CP_UTF8) != 0;
    }

    ~utf8_output_guard() noexcept {
        if (!changed_) return;
        // Flush buffered UTF-8 before restoring the invoking shell's encoding.
        try { std::cout.flush(); } catch (...) {}
        try { std::cerr.flush(); } catch (...) {}
        try { std::clog.flush(); } catch (...) {}
        std::fflush(stdout);
        std::fflush(stderr);
        if (GetConsoleOutputCP() == CP_UTF8)
            SetConsoleOutputCP(original_code_page_);
    }

    utf8_output_guard(const utf8_output_guard&) = delete;
    utf8_output_guard& operator=(const utf8_output_guard&) = delete;
};

inline void initialize_utf8_output() {
    // Static lifetime also restores the code page when help uses std::exit().
    static utf8_output_guard guard;
    (void)guard;
}
} // namespace prp_console
#else
namespace prp_console {
inline void initialize_utf8_output() {}
} // namespace prp_console
#endif
