//! Floating point k-d tree, for use when the co-ordinates of the points being stored in the tree
//! are floats. f64 or f32 are supported currently, or [`f16`](https://docs.rs/half/latest/half/struct.f16.html)
//! if the `f16` feature is enabled

use alloc::vec::Vec;
use az::{Az, Cast};
use itertools::__std_iter::Iterator;

use core::cmp::PartialEq;
use core::fmt::Debug;
use num_traits::float::FloatCore;

use crate::{
    iter::{IterableTreeData, TreeIter},
    types::{Content, Index},
};

/// Axis trait represents the traits that must be implemented
/// by the type that is used as the first generic parameter, `A`,
/// on the float [`KdTree`]. This will be [`f64`] or [`f32`],
/// or [`f16`](https://docs.rs/half/latest/half/struct.f16.html) if the `f16` feature is enabled
pub trait Axis: FloatCore + Default + Debug + Copy + Sync + Send + core::ops::AddAssign {
    /// returns absolute diff between two values of a type implementing this trait
    fn saturating_dist(self, other: Self) -> Self;

    /// used in query methods to update the rd value. Basically a saturating add for Fixed and an add for Float
    fn rd_update(rd: Self, delta: Self) -> Self;
}
impl<T: FloatCore + Default + Debug + Copy + Sync + Send + core::ops::AddAssign> Axis for T {
    fn saturating_dist(self, other: Self) -> Self {
        (self - other).abs()
    }

    #[inline]
    fn rd_update(rd: Self, delta: Self) -> Self {
        rd + delta
    }
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct StemNode<A: Copy + Default, const K: usize, IDX> {
    pub(crate) left: IDX,
    pub(crate) right: IDX,
    pub(crate) split_val: A,
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct LeafNode<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX> {
    pub content_points: [[A; K]; B],
    pub content_items: [T; B],

    pub size: IDX,
}

impl<A: Copy + Default, T: Copy + Default, const K: usize, const B: usize, IDX>
    LeafNode<A, T, K, B, IDX>
where
    A: Axis,
    T: Content,
    IDX: Index<T = IDX>,
{
    pub(crate) fn new() -> Self {
        Self {
            content_points: [[A::zero(); K]; B],
            content_items: [T::zero(); B],
            size: IDX::zero(),
        }
    }
}

macro_rules! generate_common_methods {
    ($kdtree:ident) => {
        /// Returns the current number of elements stored in the tree
        ///
        /// # Examples
        ///
        /// ```rust
        /// use kiddo::KdTree;
        ///
        /// let mut tree: KdTree<f64, 3> = KdTree::new();
        ///
        /// tree.add(&[1.0, 2.0, 5.0], 100);
        /// tree.add(&[1.1, 2.1, 5.1], 101);
        ///
        /// assert_eq!(tree.size(), 2);
        /// ```
        #[inline]
        pub fn size(&self) -> T {
            self.size
        }
    };
}
