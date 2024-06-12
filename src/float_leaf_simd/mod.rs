pub(crate) mod fallback;
pub mod leaf_node;

// TODO: fix f32 AVX2

// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx2",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// pub(crate) mod f32_avx2;

#[cfg(all(
    feature = "simd",
    target_feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub(crate) mod f64_avx2;

#[cfg(all(
    target_feature = "neon",
    any(target_arch = "aarch64", target_arch = "arm")
))]
pub(crate) mod f64_neon;

// TODO: fix f32 AVX512

// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx512f",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// pub(crate) mod f32_avx512;

// TODO: fix f64 AVX512
// #[cfg(all(
//     feature = "simd",
//     target_feature = "avx512f",
//     any(target_arch = "x86", target_arch = "x86_64")
// ))]
// pub(crate) mod f64_avx512;
