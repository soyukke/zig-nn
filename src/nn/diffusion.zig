const std = @import("std");
const math = std.math;

/// DDPM Noise Schedule (linear beta schedule)
///
/// Precomputes: betas, alphas, alpha_bars, sqrt_alpha_bars, sqrt_one_minus_alpha_bars
pub fn noise_schedule(comptime T: usize) type {
    return struct {
        const Self = @This();

        betas: [T]f32,
        alphas: [T]f32,
        alpha_bars: [T]f32,
        sqrt_alpha_bars: [T]f32,
        sqrt_one_minus_alpha_bars: [T]f32,

        pub fn init_linear(beta_start: f32, beta_end: f32) Self {
            var self: Self = undefined;
            var alpha_bar: f32 = 1.0;
            for (0..T) |t| {
                const frac = @as(f32, @floatFromInt(t)) / @as(f32, @floatFromInt(T - 1));
                self.betas[t] = beta_start + frac * (beta_end - beta_start);
                self.alphas[t] = 1.0 - self.betas[t];
                alpha_bar *= self.alphas[t];
                self.alpha_bars[t] = alpha_bar;
                self.sqrt_alpha_bars[t] = @sqrt(alpha_bar);
                self.sqrt_one_minus_alpha_bars[t] = @sqrt(1.0 - alpha_bar);
            }
            return self;
        }

        /// Sqrt schedule: ᾱ_t = 1 - √((t+1)/T + s) where s is a small offset
        /// Seq-to-seq diffusion で使用されるスケジュール
        pub fn init_sqrt(s: f32) Self {
            var self: Self = undefined;
            const ft: f32 = @floatFromInt(T);

            for (0..T) |t| {
                const frac = (@as(f32, @floatFromInt(t + 1))) / ft;
                var ab = 1.0 - @sqrt(frac + s);
                // Clamp alpha_bar to [1e-5, 0.9999]
                ab = @max(1e-5, @min(0.9999, ab));
                self.alpha_bars[t] = ab;
                self.sqrt_alpha_bars[t] = @sqrt(ab);
                self.sqrt_one_minus_alpha_bars[t] = @sqrt(1.0 - ab);

                if (t == 0) {
                    self.betas[t] = 1.0 - ab;
                } else {
                    self.betas[t] = 1.0 - ab / self.alpha_bars[t - 1];
                }
                self.betas[t] = @max(1e-5, @min(0.999, self.betas[t]));
                self.alphas[t] = 1.0 - self.betas[t];
            }
            return self;
        }

        /// Cosine schedule: ᾱ_t = cos²(((t/T + s) / (1+s)) * π/2)
        /// "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal)
        pub fn init_cosine(s: f32) Self {
            var self: Self = undefined;
            const ft: f32 = @floatFromInt(T);

            for (0..T) |t| {
                const frac = (@as(f32, @floatFromInt(t + 1))) / ft;
                const angle = ((frac + s) / (1.0 + s)) * math.pi * 0.5;
                var ab = @cos(angle) * @cos(angle);
                ab = @max(1e-5, @min(0.9999, ab));
                self.alpha_bars[t] = ab;
                self.sqrt_alpha_bars[t] = @sqrt(ab);
                self.sqrt_one_minus_alpha_bars[t] = @sqrt(1.0 - ab);

                if (t == 0) {
                    self.betas[t] = 1.0 - ab;
                } else {
                    self.betas[t] = 1.0 - ab / self.alpha_bars[t - 1];
                }
                self.betas[t] = @max(1e-5, @min(0.999, self.betas[t]));
                self.alphas[t] = 1.0 - self.betas[t];
            }
            return self;
        }

        /// Posterior mean for x_0-prediction:
        /// μ(x_t, x_0_pred, t) = (√ᾱ_{t-1} * β_t / (1-ᾱ_t)) * x_0_pred
        ///                      + (√α_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)) * x_t
        /// Posterior variance: β̃_t = β_t * (1-ᾱ_{t-1}) / (1-ᾱ_t)
        pub fn posterior_sample(
            self: *const Self,
            x_t: []const f32,
            x_0_pred: []const f32,
            x_out: []f32,
            noise: []const f32,
            t: usize,
        ) void {
            if (t == 0) {
                // Last step: just use x_0_pred
                @memcpy(x_out, x_0_pred);
                return;
            }

            const ab_t = self.alpha_bars[t];
            const ab_t_prev = self.alpha_bars[t - 1];
            const beta_t = self.betas[t];
            const alpha_t = self.alphas[t];

            // Posterior mean coefficients
            const coeff_x0 = @sqrt(ab_t_prev) * beta_t / (1.0 - ab_t);
            const coeff_xt = @sqrt(alpha_t) * (1.0 - ab_t_prev) / (1.0 - ab_t);

            // Posterior variance
            const var_t = beta_t * (1.0 - ab_t_prev) / (1.0 - ab_t);
            const sigma_t = @sqrt(var_t);

            for (x_t, x_0_pred, x_out, noise) |xt, x0p, *xo, z| {
                xo.* = coeff_x0 * x0p + coeff_xt * xt + sigma_t * z;
            }
        }
    };
}

/// Sinusoidal positional embedding for timestep
/// out[0..dim/2] = sin(t / 10000^(2i/dim))
/// out[dim/2..dim] = cos(t / 10000^(2i/dim))
pub fn sinusoidal_embedding(comptime dim: usize, timestep: usize, out: []f32) void {
    const half = dim / 2;
    const t: f32 = @floatFromInt(timestep);
    for (0..half) |i| {
        const fi: f32 = @floatFromInt(i);
        const freq = @exp(-fi * @log(@as(f32, 10000.0)) / @as(f32, @floatFromInt(half)));
        const angle = t * freq;
        out[i] = @sin(angle);
        out[half + i] = @cos(angle);
    }
}

/// Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
pub fn forward_diffusion(
    x0: []const f32,
    epsilon: []const f32,
    x_t: []f32,
    sqrt_alpha_bar: f32,
    sqrt_one_minus_alpha_bar: f32,
) void {
    for (x0, epsilon, x_t) |x, e, *xt| {
        xt.* = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * e;
    }
}

/// DDPM reverse sampling step:
/// x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_pred) + sqrt(beta_t) * z
pub fn ddpm_sample_step(
    x_t: []const f32,
    eps_pred: []const f32,
    x_out: []f32,
    noise: []const f32,
    alpha_t: f32,
    beta_t: f32,
    sqrt_one_minus_alpha_bar_t: f32,
    is_last_step: bool,
) void {
    const inv_sqrt_alpha = 1.0 / @sqrt(alpha_t);
    const coeff = beta_t / sqrt_one_minus_alpha_bar_t;
    const sigma = @sqrt(beta_t);

    for (x_t, eps_pred, x_out, noise) |xt, ep, *xo, z| {
        xo.* = inv_sqrt_alpha * (xt - coeff * ep);
        if (!is_last_step) {
            xo.* += sigma * z;
        }
    }
}

/// Box-Muller transform: generate standard normal samples from uniform random
pub fn box_muller_gaussian(rng: std.Random, out: []f32) void {
    var i: usize = 0;
    while (i + 1 < out.len) : (i += 2) {
        const uniform1 = rng.float(f32) * 0.99998 + 0.00001; // avoid log(0)
        const uniform2 = rng.float(f32);
        const r = @sqrt(-2.0 * @log(uniform1));
        const theta = 2.0 * math.pi * uniform2;
        out[i] = r * @cos(theta);
        out[i + 1] = r * @sin(theta);
    }
    // Handle odd length
    if (out.len % 2 != 0) {
        const uniform1 = rng.float(f32) * 0.99998 + 0.00001;
        const uniform2 = rng.float(f32);
        const r = @sqrt(-2.0 * @log(uniform1));
        out[out.len - 1] = r * @cos(2.0 * math.pi * uniform2);
    }
}

test "noise schedule linear" {
    const schedule = noise_schedule(10).init_linear(1e-4, 0.02);
    // beta_0 = 1e-4, beta_9 = 0.02
    try std.testing.expectApproxEqAbs(@as(f32, 1e-4), schedule.betas[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), schedule.betas[9], 1e-6);
    // alpha_bar should be decreasing
    for (1..10) |t| {
        try std.testing.expect(schedule.alpha_bars[t] < schedule.alpha_bars[t - 1]);
    }
    // sqrt consistency
    try std.testing.expectApproxEqAbs(
        @sqrt(schedule.alpha_bars[5]),
        schedule.sqrt_alpha_bars[5],
        1e-6,
    );
}

test "sinusoidal embedding" {
    var emb: [8]f32 = undefined;
    sinusoidal_embedding(8, 0, &emb);
    // sin(0) = 0 for all frequencies
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), emb[i], 1e-6);
    }
    // cos(0) = 1 for all frequencies
    for (4..8) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), emb[i], 1e-6);
    }
}

test "forward diffusion" {
    const x0 = [_]f32{ 1.0, 2.0 };
    const eps = [_]f32{ 0.5, -0.5 };
    var x_t: [2]f32 = undefined;
    forward_diffusion(&x0, &eps, &x_t, 0.9, 0.4359); // approximate values
    // x_t = 0.9 * x0 + 0.4359 * eps
    try std.testing.expectApproxEqAbs(@as(f32, 0.9 * 1.0 + 0.4359 * 0.5), x_t[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9 * 2.0 + 0.4359 * (-0.5)), x_t[1], 1e-3);
}

test "noise schedule sqrt" {
    const schedule = noise_schedule(100).init_sqrt(0.0001);
    // alpha_bar should be decreasing
    for (1..100) |t| {
        try std.testing.expect(schedule.alpha_bars[t] <= schedule.alpha_bars[t - 1]);
    }
    // alpha_bar should be in [0, 1]
    for (0..100) |t| {
        try std.testing.expect(schedule.alpha_bars[t] >= 0);
        try std.testing.expect(schedule.alpha_bars[t] <= 1);
    }
}

test "noise schedule cosine" {
    const schedule = noise_schedule(100).init_cosine(0.008);
    // alpha_bar should be decreasing
    for (1..100) |t| {
        try std.testing.expect(schedule.alpha_bars[t] <= schedule.alpha_bars[t - 1]);
    }
    // First alpha_bar should be close to 1, last close to 0
    try std.testing.expect(schedule.alpha_bars[0] > 0.9);
    try std.testing.expect(schedule.alpha_bars[99] < 0.1);
}

test "posterior sample" {
    const schedule = noise_schedule(100).init_cosine(0.008);
    const x_t = [_]f32{ 1.0, 0.5 };
    const x_0_pred = [_]f32{ 0.8, 0.3 };
    const noise = [_]f32{ 0.0, 0.0 }; // zero noise for deterministic test
    var x_out: [2]f32 = undefined;

    // t=50: should produce values between x_t and x_0_pred
    schedule.posterior_sample(&x_t, &x_0_pred, &x_out, &noise, 50);
    for (0..2) |i| {
        try std.testing.expect(!std.math.isNan(x_out[i]));
        try std.testing.expect(!std.math.isInf(x_out[i]));
    }

    // t=0: should return x_0_pred
    schedule.posterior_sample(&x_t, &x_0_pred, &x_out, &noise, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), x_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), x_out[1], 1e-6);
}

test "box muller gaussian" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();
    var samples: [1000]f32 = undefined;
    box_muller_gaussian(rng, &samples);

    // Check mean ≈ 0 and std ≈ 1
    var sum: f32 = 0;
    for (samples) |s| sum += s;
    const mean = sum / 1000.0;
    try std.testing.expect(@abs(mean) < 0.1);

    var var_sum: f32 = 0;
    for (samples) |s| var_sum += (s - mean) * (s - mean);
    const variance = var_sum / 1000.0;
    try std.testing.expect(@abs(variance - 1.0) < 0.15);
}
