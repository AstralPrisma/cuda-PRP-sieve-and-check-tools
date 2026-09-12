// SPDX-License-Identifier: GPL-2.0-or-later
// Copyright (C) 2026 AstralPrisma (A.P.).
// SHA-256 implementation reused from GSRPS 2.3.
#pragma once
#include <array>
#include <cstring>
#include <limits>
#include <algorithm>
#include <stdexcept>
class CheckpointSha256 {
public:
    CheckpointSha256()
        : _state{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void update(const uint8_t* data, size_t size) {
        if (_finalized) throw std::runtime_error("checkpoint SHA-256 update after finalization");
        if (size > std::numeric_limits<uint64_t>::max() - _total_bytes) {
            throw std::runtime_error("checkpoint SHA-256 input is too large");
        }
        _total_bytes += static_cast<uint64_t>(size);
        while (size != 0) {
            const size_t take = std::min(size, _buffer.size() - _buffer_size);
            std::memcpy(_buffer.data() + _buffer_size, data, take);
            _buffer_size += take;
            data += take;
            size -= take;
            if (_buffer_size == _buffer.size()) {
                transform(_buffer.data());
                _buffer_size = 0;
            }
        }
    }

    std::array<uint8_t, 32> final() {
        if (_finalized) throw std::runtime_error("checkpoint SHA-256 finalized twice");
        const uint64_t bit_length = _total_bytes * 8;
        _buffer[_buffer_size++] = 0x80;
        if (_buffer_size > 56) {
            std::fill(_buffer.begin() + static_cast<std::ptrdiff_t>(_buffer_size),
                      _buffer.end(), 0);
            transform(_buffer.data());
            _buffer_size = 0;
        }
        std::fill(_buffer.begin() + static_cast<std::ptrdiff_t>(_buffer_size),
                  _buffer.begin() + 56, 0);
        for (int i = 0; i < 8; ++i) {
            _buffer[63 - i] = static_cast<uint8_t>(bit_length >> (8 * i));
        }
        transform(_buffer.data());
        _finalized = true;
        std::array<uint8_t, 32> out{};
        for (size_t i = 0; i < _state.size(); ++i) {
            out[4 * i] = static_cast<uint8_t>(_state[i] >> 24);
            out[4 * i + 1] = static_cast<uint8_t>(_state[i] >> 16);
            out[4 * i + 2] = static_cast<uint8_t>(_state[i] >> 8);
            out[4 * i + 3] = static_cast<uint8_t>(_state[i]);
        }
        return out;
    }

private:
    static uint32_t rotr(uint32_t value, unsigned count) {
        return (value >> count) | (value << (32 - count));
    }

    void transform(const uint8_t block[64]) {
        static constexpr uint32_t constants[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        uint32_t schedule[64];
        for (int i = 0; i < 16; ++i) {
            schedule[i] =
                (static_cast<uint32_t>(block[4 * i]) << 24) |
                (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                static_cast<uint32_t>(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(schedule[i - 15], 7) ^
                                rotr(schedule[i - 15], 18) ^
                                (schedule[i - 15] >> 3);
            const uint32_t s1 = rotr(schedule[i - 2], 17) ^
                                rotr(schedule[i - 2], 19) ^
                                (schedule[i - 2] >> 10);
            schedule[i] = schedule[i - 16] + s0 + schedule[i - 7] + s1;
        }
        uint32_t a = _state[0], b = _state[1], c = _state[2], d = _state[3];
        uint32_t e = _state[4], f = _state[5], g = _state[6], h = _state[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t temp1 = h + s1 + choice + constants[i] + schedule[i];
            const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        _state[0] += a; _state[1] += b; _state[2] += c; _state[3] += d;
        _state[4] += e; _state[5] += f; _state[6] += g; _state[7] += h;
    }

    std::array<uint32_t, 8> _state;
    std::array<uint8_t, 64> _buffer{};
    size_t _buffer_size = 0;
    uint64_t _total_bytes = 0;
    bool _finalized = false;
};

void checkpoint_sha_u32(CheckpointSha256& sha, uint32_t value) {
    uint8_t bytes[4];
    for (int i = 0; i < 4; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    sha.update(bytes, sizeof(bytes));
}

void checkpoint_sha_u64(CheckpointSha256& sha, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    sha.update(bytes, sizeof(bytes));
}

