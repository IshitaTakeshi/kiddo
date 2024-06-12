use core::arch::aarch64::vaddq_s64;
use core::arch::aarch64::vbslq_s64;
use core::arch::aarch64::vcgeq_f64;
use core::arch::aarch64::vcgtq_f64;
use core::arch::aarch64::vcleq_f64;
use core::arch::aarch64::vdupq_n_s64;
use core::arch::aarch64::vdupq_n_u64;
use core::arch::aarch64::vgetq_lane_f64;
use core::arch::aarch64::vgetq_lane_u64;
use core::arch::aarch64::vld1q_f64;
use core::arch::aarch64::vminq_f64;
use core::arch::aarch64::vst1q_f64;
use core::arch::aarch64::vst1q_s64;

use crate::types::Content;

pub(crate) unsafe fn get_best_from_dists_f64_neon<T: Content, const B: usize>(
    acc: &[f64; B],
    items: &[T; B],
    best_dist: &mut f64,
    best_item: &mut T,
) {
    let mut index_v = vdupq_n_s64(0);
    let mut min_dist_indexes_v = vdupq_n_s64(-1);
    let all_ones = vdupq_n_s64(1);

    let mut min_dists = [*best_dist; 2];
    let mut min_dists_v = vld1q_f64(&min_dists[0] as *const f64);

    let mut any_is_better = false;

    for chunk in acc.chunks_exact(2) {
        let chunk_v = vld1q_f64(&chunk[0] as *const f64);

        let is_better_mask = vcgeq_f64(min_dists_v, chunk_v);
        let min_dist0 = vgetq_lane_f64(min_dists_v, 0);
        let min_dist1 = vgetq_lane_f64(min_dists_v, 1);

        let mask0 = vgetq_lane_u64(is_better_mask, 0);
        let mask1 = vgetq_lane_u64(is_better_mask, 1);
        any_is_better |= mask0 > 0 || mask1 > 0;

        min_dists_v = vminq_f64(min_dists_v, chunk_v);

        min_dist_indexes_v = vbslq_s64(is_better_mask, index_v, min_dist_indexes_v);

        index_v = vaddq_s64(index_v, all_ones);
    }

    if !any_is_better {
        return;
    }

    let mut min_dist_indexes = [0i64; 2];
    vst1q_s64(&mut min_dist_indexes[0], min_dist_indexes_v);
    vst1q_f64(&mut min_dists[0], min_dists_v);

    if min_dists[0] < *best_dist {
        *best_dist = min_dists[0];
        *best_item = items[min_dist_indexes[0] as usize + 0];
    }
    if min_dists[1] < *best_dist {
        *best_dist = min_dists[1];
        *best_item = items[min_dist_indexes[1] as usize + 1];
    }
}
