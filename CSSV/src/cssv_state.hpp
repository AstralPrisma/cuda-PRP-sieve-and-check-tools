// SPDX-License-Identifier: GPL-2.0-or-later
// Checked, portable CSSV state files and durable single-file publication.
#pragma once
#include "cssv_common.hpp"
#include "sha256.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace cssv {

// No floating point conversion: a sieve depth must never be silently rounded.
inline u64 parse_integer(const std::string& text) {
    if (text.empty() || text.size() > 4096)
        throw std::runtime_error("expected an unsigned integer (decimal or exact scientific notation)");
    size_t i = 0, fractional = 0;
    std::string digits;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') digits += text[i++];
    if (digits.empty()) throw std::runtime_error("expected an unsigned decimal integer");
    if (i < text.size() && text[i] == '.') {
        ++i;
        const size_t before = digits.size();
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') digits += text[i++];
        fractional = digits.size() - before;
        if (!fractional) throw std::runtime_error("missing digits after decimal point");
    }
    int exponent = 0;
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
        ++i;
        bool negative = false;
        if (i < text.size() && (text[i] == '+' || text[i] == '-')) negative = text[i++] == '-';
        const size_t begin = i;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            // A saturated exponent is sufficient to reject overflow/nonintegral
            // nonzero values, without overflowing the parser itself.
            exponent = std::min(100000, exponent * 10 + (text[i++] - '0'));
        }
        if (i == begin) throw std::runtime_error("missing scientific exponent");
        if (negative) exponent = -exponent;
    }
    if (i != text.size()) throw std::runtime_error("invalid character in unsigned integer");
    const size_t nonzero = digits.find_first_not_of('0');
    if (nonzero == std::string::npos) return 0;
    digits.erase(0, nonzero);
    const int shift = exponent - static_cast<int>(fractional);
    if (shift < 0) {
        const size_t remove = static_cast<size_t>(-shift);
        if (remove >= digits.size()) throw std::runtime_error("value is not an exact integer");
        for (size_t k = digits.size() - remove; k < digits.size(); ++k)
            if (digits[k] != '0') throw std::runtime_error("value is not an exact integer");
        digits.resize(digits.size() - remove);
    } else {
        if (digits.size() + static_cast<size_t>(shift) > 20)
            throw std::runtime_error("unsigned integer overflow");
        digits.append(static_cast<size_t>(shift), '0');
    }
    u64 result = 0;
    for (char ch : digits) {
        const unsigned digit = unsigned(ch - '0');
        if (result > (std::numeric_limits<u64>::max() - digit) / 10)
            throw std::runtime_error("unsigned integer overflow");
        result = result * 10 + digit;
    }
    return result;
}

namespace state_detail {
constexpr size_t HEADER_BYTES = 88;
constexpr size_t DIGEST_BYTES = 32;
constexpr u64 MAX_STARTS = 2000000;
constexpr u64 MAX_TERMS = 1000000;
constexpr size_t MAX_FILE_BYTES = HEADER_BYTES + size_t(MAX_STARTS) * 2 * 8 + DIGEST_BYTES;

inline size_t expected_cells(const Config& c) {
    if (!c.start || c.end < c.start || c.end - c.start >= MAX_STARTS)
        throw std::runtime_error("invalid CSSV state starting-value range");
    if (!c.terms || c.terms > MAX_TERMS || (c.step != 1 && c.step != 2))
        throw std::runtime_error("invalid CSSV state terms or step");
    if (c.sign != -1 && c.sign != 1)
        throw std::runtime_error("invalid CSSV state sign");
    const u64 extra = (c.terms - 1) * c.step;
    if (c.end > std::numeric_limits<u64>::max() - extra - 1)
        throw std::runtime_error("CSSV final base or exponent overflows uint64");
    if (c.pmin < 2 || c.pmin > MAX_P || c.pmax < c.pmin || c.pmax > MAX_P)
        throw std::runtime_error("invalid CSSV state sieve bounds");
    return static_cast<size_t>(c.end - c.start + 1) * (c.both ? 2 : 1);
}

inline void validate(const Snapshot& state) {
    const size_t cells = expected_cells(state.cfg);
    if (state.frontier < state.cfg.pmin - 1 || state.frontier > state.cfg.pmax)
        throw std::runtime_error("CSSV state frontier lies outside its declared sieve range");
    if (state.factors.size() != cells) throw std::runtime_error("CSSV state factor count mismatch");
    for (u64 p : state.factors)
        if (p != NONE && (p < state.cfg.pmin || p > state.frontier))
            throw std::runtime_error("CSSV state contains a factor outside its completed sieve range");
}

inline void put32(std::vector<uint8_t>& bytes, uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) bytes.push_back(uint8_t(value >> (i * 8)));
}
inline void put64(std::vector<uint8_t>& bytes, u64 value) {
    for (unsigned i = 0; i < 8; ++i) bytes.push_back(uint8_t(value >> (i * 8)));
}
inline uint32_t get32(const std::vector<uint8_t>& bytes, size_t& at) {
    if (at > bytes.size() || bytes.size() - at < 4) throw std::runtime_error("truncated CSSV state");
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= uint32_t(bytes[at++]) << (i * 8);
    return value;
}
inline u64 get64(const std::vector<uint8_t>& bytes, size_t& at) {
    if (at > bytes.size() || bytes.size() - at < 8) throw std::runtime_error("truncated CSSV state");
    u64 value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= u64(bytes[at++]) << (i * 8);
    return value;
}

inline std::runtime_error file_error(const char* operation, const std::filesystem::path& path,
                                    unsigned long code) {
    return std::runtime_error(std::string(operation) + ": " + path.u8string() +
                              " (OS error " + std::to_string(code) + ")");
}

// Temporary file is in the destination directory, exclusively created, flushed,
// then atomically renamed. A failure before publication leaves the old file intact.
inline void atomic_write(const std::filesystem::path& destination, const uint8_t* bytes, size_t size) {
    if (destination.empty() || destination.filename().empty()) throw std::runtime_error("empty output path");
    static std::atomic<unsigned long long> serial{0};
    std::filesystem::path temporary;
#ifdef _WIN32
    HANDLE fd = INVALID_HANDLE_VALUE;
    for (unsigned attempt = 0; attempt < 100 && fd == INVALID_HANDLE_VALUE; ++attempt) {
        temporary = destination;
        temporary += ".tmp." + std::to_string(GetCurrentProcessId()) + "." + std::to_string(++serial);
        fd = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fd == INVALID_HANDLE_VALUE && GetLastError() != ERROR_FILE_EXISTS && GetLastError() != ERROR_ALREADY_EXISTS)
            throw file_error("cannot create temporary output", temporary, GetLastError());
    }
    if (fd == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot reserve a unique temporary output");
    try {
        while (size) {
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(size, 1u << 24));
            DWORD written = 0;
            if (!WriteFile(fd, bytes, chunk, &written, nullptr) || !written)
                throw file_error("cannot write temporary output", temporary, GetLastError());
            bytes += written;
            size -= written;
        }
        if (!FlushFileBuffers(fd)) throw file_error("cannot flush temporary output", temporary, GetLastError());
        if (!CloseHandle(fd)) {
            fd = INVALID_HANDLE_VALUE;
            throw file_error("cannot close temporary output", temporary, GetLastError());
        }
        fd = INVALID_HANDLE_VALUE;
        if (!MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw file_error("cannot atomically publish output", destination, GetLastError());
    } catch (...) {
        if (fd != INVALID_HANDLE_VALUE) CloseHandle(fd);
        DeleteFileW(temporary.c_str());
        throw;
    }
#else
    int fd = -1;
    for (unsigned attempt = 0; attempt < 100 && fd < 0; ++attempt) {
        temporary = destination;
        temporary += ".tmp." + std::to_string(static_cast<unsigned long long>(getpid())) + "." + std::to_string(++serial);
        fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0 && errno != EEXIST) throw file_error("cannot create temporary output", temporary, errno);
    }
    if (fd < 0) throw std::runtime_error("cannot reserve a unique temporary output");
    int directory_fd = -1;
    bool published = false;
    try {
        const auto parent = destination.has_parent_path() ? destination.parent_path() : std::filesystem::path(".");
        directory_fd = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (directory_fd < 0) throw file_error("cannot open output directory", parent, errno);
        while (size) {
            const ssize_t written = ::write(fd, bytes, std::min<size_t>(size, 1u << 24));
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw file_error("cannot write temporary output", temporary, errno);
            bytes += written;
            size -= static_cast<size_t>(written);
        }
        if (::fsync(fd) != 0) throw file_error("cannot fsync temporary output", temporary, errno);
        if (::close(fd) != 0) {
            fd = -1;
            throw file_error("cannot close temporary output", temporary, errno);
        }
        fd = -1;
        if (::rename(temporary.c_str(), destination.c_str()) != 0)
            throw file_error("cannot atomically publish output", destination, errno);
        published = true;
        if (::fsync(directory_fd) != 0)
            throw file_error("output was replaced, but directory fsync failed", parent, errno);
        ::close(directory_fd);
        directory_fd = -1;
    } catch (...) {
        if (fd >= 0) ::close(fd);
        if (directory_fd >= 0) ::close(directory_fd);
        if (!published) ::unlink(temporary.c_str());
        throw;
    }
#endif
}
}  // namespace state_detail

inline void save_state(const std::string& path, const Snapshot& state) {
    state_detail::validate(state);
    std::vector<uint8_t> bytes;
    bytes.reserve(state_detail::HEADER_BYTES + state.factors.size() * 8 + state_detail::DIGEST_BYTES);
    const char magic[] = "CSSVST01";
    bytes.insert(bytes.end(), magic, magic + 8);
    state_detail::put32(bytes, 1);  // schema version
    state_detail::put32(bytes, 0);  // reserved; reject unknown extensions
    const Config& c = state.cfg;
    for (u64 value : {c.start, c.end, c.terms, c.pmin, c.pmax, state.frontier, u64(state.factors.size())})
        state_detail::put64(bytes, value);
    state_detail::put32(bytes, c.step);
    state_detail::put32(bytes, c.sign == -1 ? 0 : 1);
    state_detail::put32(bytes, c.both ? 1 : 0);
    state_detail::put32(bytes, c.reference ? 1 : 0);
    if (bytes.size() != state_detail::HEADER_BYTES) throw std::runtime_error("internal CSSV state schema mismatch");
    for (u64 p : state.factors) state_detail::put64(bytes, p);
    CheckpointSha256 hash;
    hash.update(bytes.data(), bytes.size());
    const auto digest = hash.final();
    bytes.insert(bytes.end(), digest.begin(), digest.end());
    state_detail::atomic_write(std::filesystem::u8path(path), bytes.data(), bytes.size());
}

inline Snapshot load_state(const std::string& path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open CSSV state: " + path);
    const auto length = input.tellg();
    if (length < std::streamoff(state_detail::HEADER_BYTES + state_detail::DIGEST_BYTES) ||
        length > std::streamoff(state_detail::MAX_FILE_BYTES))
        throw std::runtime_error("invalid CSSV state length");
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input || input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("CSSV state read failed or file changed while reading");
    if (std::memcmp(bytes.data(), "CSSVST01", 8) != 0) throw std::runtime_error("unrecognized CSSV state magic");
    CheckpointSha256 hash;
    hash.update(bytes.data(), bytes.size() - state_detail::DIGEST_BYTES);
    const auto digest = hash.final();
    unsigned different = 0;
    for (size_t i = 0; i < digest.size(); ++i) different |= digest[i] ^ bytes[bytes.size() - digest.size() + i];
    if (different) throw std::runtime_error("CSSV state SHA-256 mismatch");
    size_t at = 8;
    if (state_detail::get32(bytes, at) != 1 || state_detail::get32(bytes, at) != 0)
        throw std::runtime_error("unsupported CSSV state schema");
    Snapshot state;
    Config& c = state.cfg;
    c.start = state_detail::get64(bytes, at);
    c.end = state_detail::get64(bytes, at);
    c.terms = state_detail::get64(bytes, at);
    c.pmin = state_detail::get64(bytes, at);
    c.pmax = state_detail::get64(bytes, at);
    state.frontier = state_detail::get64(bytes, at);
    const u64 count = state_detail::get64(bytes, at);
    c.step = state_detail::get32(bytes, at);
    const uint32_t sign = state_detail::get32(bytes, at);
    const uint32_t both = state_detail::get32(bytes, at);
    const uint32_t reference = state_detail::get32(bytes, at);
    if (sign > 1 || both > 1 || reference > 1) throw std::runtime_error("invalid CSSV state flags");
    c.sign = sign ? 1 : -1;
    c.both = both != 0;
    c.reference = reference != 0;
    const size_t expected = state_detail::expected_cells(c);
    if (count != expected || at != state_detail::HEADER_BYTES ||
        bytes.size() != at + expected * 8 + state_detail::DIGEST_BYTES)
        throw std::runtime_error("CSSV state dimensions or length mismatch");
    state.factors.resize(expected);
    for (u64& p : state.factors) p = state_detail::get64(bytes, at);
    state_detail::validate(state);
    return state;
}

inline void write_factors(const std::string& path, const Snapshot& state) {
    state_detail::validate(state);
    const size_t width = state.cfg.both ? 2 : 1;
    std::string text;
    // Avoid reserving 80 MB for sparse output, but bound any growth by 84 MB.
    text.reserve(state.factors.size() * 3);
    for (size_t i = 0; i < state.factors.size(); i += width) {
        for (size_t j = 0; j < width; ++j) {
            if (j) text += ' ';
            const u64 p = state.factors[i + j];
            text += std::to_string(p == NONE ? 0 : p);
        }
        text += '\n';
    }
    state_detail::atomic_write(std::filesystem::u8path(path), reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

// An OS lock, not a stale-PID file. Keep the sidecar after unlocking so another
// process can never acquire a different inode under the same lock filename.
class PathLock {
public:
    explicit PathLock(const std::string& output_path) {
        auto path = std::filesystem::u8path(output_path);
        if (path.empty() || path.filename().empty()) throw std::runtime_error("empty locked output path");
        path += ".lock";
#ifdef _WIN32
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ == INVALID_HANDLE_VALUE)
            throw state_detail::file_error("output is locked or lock cannot be opened", path, GetLastError());
#else
        fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd_ < 0) throw state_detail::file_error("cannot open output lock", path, errno);
        if (::flock(fd_, LOCK_EX | LOCK_NB) != 0) {
            const int saved = errno;
            ::close(fd_);
            fd_ = -1;
            throw state_detail::file_error("output is locked by another process", path, saved);
        }
#endif
    }
    ~PathLock() {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
#else
        if (fd_ >= 0) { ::flock(fd_, LOCK_UN); ::close(fd_); }
#endif
    }
    PathLock(const PathLock&) = delete;
    PathLock& operator=(const PathLock&) = delete;
    PathLock(PathLock&&) = delete;
    PathLock& operator=(PathLock&&) = delete;
private:
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};
}  // namespace cssv
