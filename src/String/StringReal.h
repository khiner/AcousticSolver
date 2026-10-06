#pragma once

// A float pair retains about 48 significand bits on Metal. Explicit FMA residuals
// require compiling with fast math disabled. The host only packs/unpacks doubles.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#endif
struct StringReal {
    float Hi{}, Lo{};
    StringReal() = default;
    StringReal(float value) : Hi(value) {}
    StringReal(float hi, float lo) : Hi(hi), Lo(lo) {}
};
#ifdef __METAL_VERSION__
inline StringReal StringNormalize(float hi, float lo) {
    float sum = hi + lo;
    return {sum, lo - (sum - hi)};
}
inline StringReal operator+(StringReal a, StringReal b) {
    float sum = a.Hi + b.Hi, v = sum - a.Hi;
    float error = ((b.Hi - v) + (a.Hi - (sum - v))) + a.Lo + b.Lo;
    return StringNormalize(sum, error);
}
inline StringReal operator-(StringReal a) { return {-a.Hi, -a.Lo}; }
inline StringReal operator-(StringReal a, StringReal b) { return a + -b; }
inline StringReal operator*(StringReal a, StringReal b) {
    float product = a.Hi * b.Hi;
    float error = metal::fma(a.Hi, b.Hi, -product) + a.Hi * b.Lo + a.Lo * b.Hi;
    return StringNormalize(product, error + a.Lo * b.Lo);
}
inline StringReal operator/(StringReal a, StringReal b) {
    float q = a.Hi / b.Hi;
    StringReal residual = a - b * StringReal(q);
    return StringReal(q) + StringReal((residual.Hi + residual.Lo) / b.Hi);
}
inline StringReal StringSqrt(StringReal a) {
    if (a.Hi == 0.0f) return {};
    StringReal root(metal::sqrt(a.Hi));
    return root + (a - root * root) / (StringReal(2) * root);
}
#endif
