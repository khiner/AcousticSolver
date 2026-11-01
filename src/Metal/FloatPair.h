#pragma once

// A float pair retains about 48 significand bits on Metal; explicit FMA residuals require compiling with fast math disabled.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#endif
struct FloatPair {
    float Hi{}, Lo{};
    FloatPair() = default;
    FloatPair(float value) : Hi(value) {}
    FloatPair(float hi, float lo) : Hi(hi), Lo(lo) {}
};
#ifdef __METAL_VERSION__
inline FloatPair FloatPairNormalize(float hi, float lo) {
    float sum = hi + lo;
    return {sum, lo - (sum - hi)};
}
inline FloatPair operator+(FloatPair a, FloatPair b) {
    float sum = a.Hi + b.Hi, v = sum - a.Hi;
    float error = ((b.Hi - v) + (a.Hi - (sum - v))) + a.Lo + b.Lo;
    return FloatPairNormalize(sum, error);
}
inline FloatPair operator-(FloatPair a) { return {-a.Hi, -a.Lo}; }
inline FloatPair operator-(FloatPair a, FloatPair b) { return a + -b; }
inline FloatPair operator*(FloatPair a, FloatPair b) {
    float product = a.Hi * b.Hi;
    float error = metal::fma(a.Hi, b.Hi, -product) + a.Hi * b.Lo + a.Lo * b.Hi;
    return FloatPairNormalize(product, error + a.Lo * b.Lo);
}
inline FloatPair operator/(FloatPair a, FloatPair b) {
    float q = a.Hi / b.Hi;
    FloatPair residual = a - b * FloatPair(q);
    return FloatPair(q) + FloatPair((residual.Hi + residual.Lo) / b.Hi);
}
inline FloatPair FloatPairSqrt(FloatPair a) {
    if (a.Hi == 0.0f) return {};
    FloatPair root(metal::sqrt(a.Hi));
    return root + (a - root * root) / (FloatPair(2) * root);
}
inline FloatPair FloatPairSign(FloatPair a) { return FloatPair(a.Hi > 0 ? 1.f : (a.Hi < 0 ? -1.f : 0.f)); }
#endif
