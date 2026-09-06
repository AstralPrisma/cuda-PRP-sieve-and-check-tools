/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace gfnsv_efficiency {

struct Config {
    double max_factor_seconds = 0.0;
    double max_average_factor_seconds = 0.0;
    double window_minutes = 0.0;

    void validate() const {
        if (max_factor_seconds == 0.0 && max_average_factor_seconds == 0.0 &&
            window_minutes == 0.0) return;
        if (!std::isfinite(max_factor_seconds) || max_factor_seconds <= 0.0 ||
            !std::isfinite(max_average_factor_seconds) || max_average_factor_seconds <= 0.0 ||
            !std::isfinite(window_minutes) || window_minutes <= 0.0 ||
            !std::isfinite(window_minutes * 60.0))
            throw std::invalid_argument("efficiency limits require all three finite positive values and a finite window in seconds");
    }
};

struct Status {
    bool stop = false;
    bool window_ready = false;
    std::uint64_t window_removed = 0;
    double idle_seconds = 0.0;
    double average_seconds_per_factor = 0.0;
    std::string reason;
};

class Monitor {
public:
    explicit Monitor(Config config, double start_seconds = 0.0)
        : config_(config), start_(start_seconds), last_observed_(start_seconds),
          last_removal_(start_seconds) {
        config_.validate();
        if (!std::isfinite(start_seconds))
            throw std::invalid_argument("efficiency start time must be finite");
        enabled_ = config_.max_factor_seconds > 0.0;
        window_seconds_ = config_.window_minutes * 60.0;
    }

    bool enabled() const { return enabled_; }

    Status observe(double now_seconds, std::uint64_t newly_removed) {
        if (!std::isfinite(now_seconds) || now_seconds < last_observed_)
            throw std::invalid_argument("efficiency observation time must be finite and monotone");
        Status status;
        if (!enabled_) {
            last_observed_ = now_seconds;
            return status;
        }

        // Keep the closed window [now-window, now]. Check the retained sum
        // before committing changes so rejected observations do not alter it.
        std::size_t expired = 0;
        std::uint64_t retained = window_removed_;
        for (const Sample& sample : samples_) {
            if (now_seconds - sample.when <= window_seconds_) break;
            retained -= sample.removed;
            ++expired;
        }
        if (newly_removed > (std::numeric_limits<std::uint64_t>::max)() - retained)
            throw std::overflow_error("efficiency window removal count overflow");
        if (newly_removed != 0) samples_.push_back({now_seconds, newly_removed});
        while (expired != 0) {
            samples_.pop_front();
            --expired;
        }
        window_removed_ = retained + newly_removed;
        last_observed_ = now_seconds;
        if (newly_removed != 0) last_removal_ = now_seconds;

        status.window_removed = window_removed_;
        status.idle_seconds = now_seconds - last_removal_;
        status.window_ready = now_seconds - start_ >= window_seconds_;
        status.average_seconds_per_factor = window_removed_ == 0
            ? (std::numeric_limits<double>::infinity)()
            : window_seconds_ / static_cast<double>(window_removed_);

        // The idle limit also applies before the average window is ready.
        if (status.idle_seconds > config_.max_factor_seconds) {
            status.stop = true;
            std::ostringstream reason;
            reason << "no newly removed term for " << std::fixed << std::setprecision(3)
                   << status.idle_seconds << "s, limit=" << std::defaultfloat
                   << std::setprecision(6) << config_.max_factor_seconds << "s";
            status.reason = reason.str();
        } else if (status.window_ready &&
                   status.average_seconds_per_factor > config_.max_average_factor_seconds) {
            status.stop = true;
            std::ostringstream reason;
            reason << "rolling " << std::setprecision(6) << config_.window_minutes
                   << "m window removed " << status.window_removed << " term(s), average=";
            if (std::isfinite(status.average_seconds_per_factor))
                reason << status.average_seconds_per_factor << "s/factor";
            else
                reason << "infinite s/factor";
            reason << ", limit=" << config_.max_average_factor_seconds << "s/factor";
            status.reason = reason.str();
        }
        return status;
    }

private:
    struct Sample {
        double when;
        std::uint64_t removed;
    };

    Config config_;
    bool enabled_ = false;
    double window_seconds_ = 0.0;
    double start_ = 0.0;
    double last_observed_ = 0.0;
    double last_removal_ = 0.0;
    std::deque<Sample> samples_;
    std::uint64_t window_removed_ = 0;
};

} // namespace gfnsv_efficiency
