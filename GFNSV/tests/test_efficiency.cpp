/*
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Copyright (C) 2026 AstralPrisma (A.P.)
 */
#include "../src/gfnsv_efficiency.hpp"

#include <iostream>

using gfnsv_efficiency::Config;
using gfnsv_efficiency::Monitor;

static unsigned checks = 0;
static void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class Exception, class Function>
static void rejects(Function function, const char* message) {
    bool caught = false;
    try { function(); } catch (const Exception&) { caught = true; }
    check(caught, message);
}

static void test_configuration() {
    Config{}.validate();
    Monitor disabled(Config{});
    check(!disabled.enabled(), "all-zero configuration must be disabled");
    auto status = disabled.observe(1000.0, (std::numeric_limits<std::uint64_t>::max)());
    check(!status.stop && !status.window_ready && status.window_removed == 0,
          "disabled monitor must not collect removals or stop");
    check(!disabled.observe(1000.0, 1).stop, "disabled repeated time must work");

    for (unsigned mask = 1; mask < 7; ++mask) {
        Config config{mask & 1 ? 1.0 : 0.0, mask & 2 ? 1.0 : 0.0,
                      mask & 4 ? 1.0 : 0.0};
        rejects<std::invalid_argument>([&] { config.validate(); }, "partial configuration accepted");
    }
    const double invalid[] = {-1.0, (std::numeric_limits<double>::quiet_NaN)(),
                              (std::numeric_limits<double>::infinity)(),
                              -(std::numeric_limits<double>::infinity)()};
    for (double value : invalid) {
        rejects<std::invalid_argument>([&] { Config{value, 1, 1}.validate(); }, "invalid idle limit accepted");
        rejects<std::invalid_argument>([&] { Config{1, value, 1}.validate(); }, "invalid average limit accepted");
        rejects<std::invalid_argument>([&] { Config{1, 1, value}.validate(); }, "invalid window accepted");
    }
    const double huge = (std::numeric_limits<double>::max)();
    rejects<std::invalid_argument>([&] { Config{1, 1, huge}.validate(); }, "overflowing minute conversion accepted");
    const double large_minutes = huge / 120.0;
    const double large_seconds = large_minutes * 60.0;
    Monitor large_window(Config{huge, huge, large_minutes});
    status = large_window.observe(large_seconds / 2.0, 1);
    check(!status.window_ready && !status.stop, "large finite window must not use a narrowing time cast");
    check(large_window.observe(large_seconds, 1).window_ready, "large finite window must become ready");
}

static void test_idle_and_warmup() {
    Monitor monitor(Config{10.0, 1000.0, 1.0});
    auto status = monitor.observe(10.0, 0);
    check(status.idle_seconds == 10.0 && !status.stop && !status.window_ready,
          "idle threshold equality must not stop");
    status = monitor.observe(std::nextafter(10.0, 11.0), 0);
    check(status.stop && !status.window_ready && status.reason.find("no newly removed") != std::string::npos,
          "idle timeout must apply during warmup");
    status = monitor.observe(20.0, 1);
    check(!status.stop && status.idle_seconds == 0.0 && status.window_removed == 1,
          "new removal must reset idle time");
    check(!monitor.observe(30.0, 0).stop, "reset idle threshold equality must not stop");

    Monitor warmup(Config{1000.0, 0.5, 1.0});
    status = warmup.observe(std::nextafter(60.0, 0.0), 1);
    check(!status.window_ready && !status.stop, "average limit must wait for the full window");
    status = warmup.observe(60.0, 0);
    check(status.window_ready && status.stop && status.average_seconds_per_factor == 60.0,
          "average must be checked exactly when the window becomes full");
}

static void test_window_boundaries() {
    Monitor monitor(Config{1000.0, 30.0, 1.0});
    monitor.observe(0.0, 1);
    monitor.observe(30.0, 1);
    auto status = monitor.observe(60.0, 0);
    check(status.window_ready && status.window_removed == 2 && !status.stop &&
          status.average_seconds_per_factor == 30.0,
          "closed window and average threshold equality must be retained");
    status = monitor.observe(std::nextafter(60.0, 61.0), 0);
    check(status.window_removed == 1 && status.stop && status.average_seconds_per_factor == 60.0,
          "sample must expire immediately outside the window");
    status = monitor.observe(90.0, 0);
    check(status.window_removed == 1, "sample at exactly the old boundary must remain");
    status = monitor.observe(std::nextafter(90.0, 91.0), 0);
    check(status.window_removed == 0 && std::isinf(status.average_seconds_per_factor) && status.stop,
          "empty full window must have infinite seconds per removal");

    Monitor repeated(Config{1000.0, 1.0, 1.0});
    repeated.observe(5.0, 20);
    repeated.observe(5.0, 30);
    status = repeated.observe(5.0, 10);
    check(status.window_removed == 60 && status.idle_seconds == 0.0,
          "positive batches with repeated timestamps must all count");
    check(!repeated.observe(60.0, 0).stop, "repeated-time removals must give the exact average");
    check(repeated.observe(65.0, 0).window_removed == 60, "all repeated-time boundary samples must remain");
    check(repeated.observe(std::nextafter(65.0, 66.0), 0).window_removed == 0,
          "all repeated-time expired samples must be removed");

    Monitor many(Config{1000.0, 1000.0, 1.0});
    for (unsigned i = 0; i < 10000; ++i) many.observe(static_cast<double>(i) / 1000.0, 1);
    check(many.observe(60.0, 0).window_removed == 10000,
          "rolling samples must not be truncated by a fixed sample count");
    check(many.observe(70.0, 0).window_removed == 0, "all stale samples must expire");
}

static void test_time_and_overflow() {
    const double infinity = (std::numeric_limits<double>::infinity)();
    const double nan = (std::numeric_limits<double>::quiet_NaN)();
    const Config config{1000.0, 1000.0, 1.0};
    rejects<std::invalid_argument>([&] { Monitor invalid(config, infinity); }, "infinite start accepted");
    rejects<std::invalid_argument>([&] { Monitor invalid(config, nan); }, "NaN start accepted");
    Monitor monitor(config, 100.0);
    rejects<std::invalid_argument>([&] { monitor.observe(99.0, 1); }, "time before start accepted");
    monitor.observe(110.0, 2);
    rejects<std::invalid_argument>([&] { monitor.observe(109.0, 1); }, "time rollback accepted");
    rejects<std::invalid_argument>([&] { monitor.observe(infinity, 1); }, "infinite observation accepted");
    rejects<std::invalid_argument>([&] { monitor.observe(nan, 1); }, "NaN observation accepted");
    auto status = monitor.observe(110.0, 0);
    check(status.window_removed == 2 && status.idle_seconds == 0.0,
          "invalid observations must leave monitor state unchanged");
    status = monitor.observe(160.0, 0);
    check(status.window_ready && status.window_removed == 2, "nonzero start must govern window warmup");

    const std::uint64_t maximum = (std::numeric_limits<std::uint64_t>::max)();
    Monitor overflow(config);
    overflow.observe(0.0, maximum);
    rejects<std::overflow_error>([&] { overflow.observe(1.0, 1); }, "window count overflow accepted");
    status = overflow.observe(0.0, 0);
    check(status.window_removed == maximum, "overflow must not mutate time or counters");
    rejects<std::overflow_error>([&] { overflow.observe(60.0, 1); }, "boundary count overflow accepted");
    status = overflow.observe(std::nextafter(60.0, 61.0), 1);
    check(status.window_removed == 1, "expired counts must not cause false overflow");
}

int main() {
    try {
        test_configuration();
        test_idle_and_warmup();
        test_window_boundaries();
        test_time_and_overflow();
        std::cout << "efficiency selftest ok: " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "efficiency selftest failed: " << error.what() << '\n';
        return 1;
    }
}
