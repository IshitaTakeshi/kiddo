#![allow(clippy::type_complexity, clippy::unit_arg)]
use alloc::vec;
use alloc::vec::Vec;
use alloc::boxed::Box;

use az::Cast;
use rand::distributions::{Distribution, Standard};
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelRefIterator;
use core::array;
use core::hint::black_box;

use crate::float::kdtree::{Axis, KdTree};
use crate::float_leaf_simd::leaf_node::BestFromDists;
use crate::immutable::float::kdtree::ImmutableKdTree;
use crate::types::{Content, Index};

// use rand_distr::UnitSphere as SPHERE;

/*fn rand_unit_sphere_point_f64() -> [f64; 3] {
    SPHERE.sample(&mut rand::thread_rng())
}

fn rand_sphere_data() -> ([f64; 3], usize) {
    (rand_unit_sphere_point_f64(), rand::random())
}*/

#[doc(hidden)]
#[macro_export]
macro_rules! size_t_idx {
    ( $group:ident; $callee:ident; $a:ty|$k:tt; [$(($size:tt,$t:ty,$idx:ty)),+] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, concat!($k, "D ", stringify!($a)));)* }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! batch_benches {
    ($group:ident, $callee:ident, [$(($a:ty, $k:tt)),+], $s_t_idx_list:tt ) => {
        { $($crate::size_t_idx!($group; $callee; $a|$k; $s_t_idx_list );)* }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! size_t_idx_parameterized {
    ( $group:ident; $callee:ident; $param:tt;  $a:ty|$k:tt; [$(($size:tt,$t:ty,$idx:ty)),+] ) => {
        { $($callee!($group, $a, $t, $k, $idx, $size, $param, concat!($k, "D ", stringify!($a)));)* }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! batch_benches_parameterized {
    ($group:ident, $callee:ident, $param:tt,  [$(($a:ty, $k:tt)),+], $s_t_idx_list:tt ) => {
        { $($crate::size_t_idx_parameterized!($group; $callee; $param; $a|$k; $s_t_idx_list );)* }
    }
}

pub fn build_populated_tree_float<
    A: Axis,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    spare_capacity: usize,
) -> KdTree<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<([A; K], T)>,
    Standard: Distribution<[A; K]>,
{
    let mut kdtree = KdTree::<A, T, K, B, IDX>::with_capacity(size + spare_capacity);

    for _ in 0..size {
        let entry = rand::random::<([A; K], T)>();
        kdtree.add(&entry.0, entry.1);
    }

    kdtree
}

pub fn build_populated_tree_immutable_float<A: Axis, T: Content, const K: usize, const B: usize>(
    size: usize,
) -> ImmutableKdTree<A, T, K, B>
where
    usize: Cast<T>,
    Standard: Distribution<T>,
    Standard: Distribution<([A; K], T)>,
    Standard: Distribution<[A; K]>,
    A: BestFromDists<T, B>,
{
    let mut points = vec![];
    points.resize_with(size, rand::random::<[A; K]>);

    ImmutableKdTree::<A, T, K, B>::new_from_slice(&points)
}

/*
pub fn build_populated_tree_float_sss<
    A: AxisSSS,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    spare_capacity: usize,
) -> KdTreeSSS<A, T, K, B, IDX>
where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<([A; K], T)>,
    Standard: Distribution<[A; K]>,
{
    let mut kdtree = KdTreeSSS::<A, T, K, B, IDX>::with_capacity(size + spare_capacity);

    for _ in 0..size {
        let entry = rand::random::<([A; K], T)>();
        kdtree.add(&entry.0, entry.1);
    }

    kdtree
}
*/

pub fn build_query_points_float<A: Axis, const K: usize>(points_qty: usize) -> Vec<[A; K]>
where
    Standard: Distribution<[A; K]>,
{
    (0..points_qty).map(|_| rand::random::<[A; K]>()).collect()
}

pub fn build_populated_tree_and_query_points_float<
    A: Axis,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    query_point_qty: usize,
) -> (KdTree<A, T, K, B, IDX>, Vec<[A; K]>)
where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    (
        build_populated_tree_float(size, 0),
        build_query_points_float(query_point_qty),
    )
}

pub fn build_populated_tree_and_query_points_immutable_float<
    A: Axis,
    T: Content,
    const K: usize,
    const B: usize,
>(
    size: usize,
    query_point_qty: usize,
) -> (ImmutableKdTree<A, T, K, B>, Vec<[A; K]>)
where
    A: BestFromDists<T, B>,
    usize: Cast<T>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    (
        build_populated_tree_immutable_float(size),
        build_query_points_float(query_point_qty),
    )
}

/*
pub fn build_populated_tree_and_query_points_float_sss<
    A: AxisSSS,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
>(
    size: usize,
    query_point_qty: usize,
) -> (KdTreeSSS<A, T, K, B, IDX>, Vec<[A; K]>)
where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
{
    (
        build_populated_tree_float_sss(size, 0),
        build_query_points_float(query_point_qty),
    )
}
*/

#[inline]
pub fn process_queries_float<
    A: Axis + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
) -> Box<dyn Fn((KdTree<A, T, K, B, IDX>, Vec<[A; K]>))>
where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
    F: Fn(&KdTree<A, T, K, B, IDX>, &[A; K]) + 'static + Sync,
{
    Box::new(
        move |(kdtree, points_to_query): (KdTree<A, T, K, B, IDX>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .par_iter()
                    .for_each(|point| black_box(query(&kdtree, point))),
            )
        },
    )
}

#[inline]
pub fn process_queries_immutable_float<
    A: Axis + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    F,
>(
    query: F,
) -> Box<dyn Fn((ImmutableKdTree<A, T, K, B>, Vec<[A; K]>))>
where
    usize: Cast<T>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
    F: Fn(&ImmutableKdTree<A, T, K, B>, &[A; K]) + 'static + Sync,
{
    Box::new(
        move |(kdtree, points_to_query): (ImmutableKdTree<A, T, K, B>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .par_iter()
                    .for_each(|point| black_box(query(&kdtree, point))),
            )
        },
    )
}

/*
#[inline]
pub fn process_queries_float_sss<
    A: AxisSSS + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
) -> Box<dyn Fn((KdTreeSSS<A, T, K, B, IDX>, Vec<[A; K]>))>
where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
    F: Fn(&KdTreeSSS<A, T, K, B, IDX>, &[A; K]) + 'static,
{
    Box::new(
        move |(kdtree, points_to_query): (KdTreeSSS<A, T, K, B, IDX>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .iter()
                    .for_each(|point| black_box(query(&kdtree, point))),
            )
        },
    )
}
*/

pub fn process_queries_float_parameterized<
    A: Axis + 'static,
    T: Content,
    const K: usize,
    const B: usize,
    IDX: Index<T = IDX>,
    F,
>(
    query: F,
    param: f64,
) -> Box<dyn Fn((KdTree<A, T, K, B, IDX>, Vec<[A; K]>))>
where
    usize: Cast<IDX>,
    Standard: Distribution<T>,
    Standard: Distribution<[A; K]>,
    F: Fn(&KdTree<A, T, K, B, IDX>, &[A; K], f64) + 'static,
{
    Box::new(
        move |(kdtree, points_to_query): (KdTree<A, T, K, B, IDX>, Vec<[A; K]>)| {
            black_box(
                points_to_query
                    .iter()
                    .for_each(|point| black_box(query(&kdtree, point, param))),
            )
        },
    )
}
