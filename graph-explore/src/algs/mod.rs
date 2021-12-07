use std::fmt::Debug;

use crate::cost::Cost;
use crate::graph::{Graph, Node};

mod blocking;
mod dfs;
mod hdfs;
mod robust;
mod salesperson;

pub use blocking::{blocking, pblocking, Blocking, BlockingType};
pub use dfs::{dfs, pdfs, DFSType, DepthFirstSearch};
pub use hdfs::{hdfs, hdfs_rounding, HierarchicalDFS};
pub use robust::*;
pub use salesperson::{salesperson, Salesperson};

#[derive(Debug, Clone)]
pub struct Move {
    pub current: Node,
    pub next: Node,
    pub cost: Cost,
}

#[derive(Debug, Clone)]
pub enum Action {
    Move(Move),
    Finished,
}

impl Action {
    pub fn is_move(&self) -> bool {
        matches!(self, Action::Move(_))
    }

    pub fn next_move(self) -> Option<Move> {
        if let Action::Move(m) = self {
            Some(m)
        } else {
            None
        }
    }

    pub fn is_finished(&self) -> bool {
        matches!(self, Action::Finished)
    }
}

/// Defines a deterministic algorithm for the graph exploration problem.
pub trait DetAlgorithm<'a, G>: Debug + Clone
where
    G: Graph<'a>,
{
    /// Returns the next action of the algorithm if there is any without executing it.
    ///
    /// The internal state will **not** be modified by calling this method. Consecutive calls of this operation without
    /// calling `explore_next` must always return the same value.
    fn peek_next(&mut self) -> Action;

    /// Executes the next action of the algorithm and modifies the internal state.
    ///
    /// The node which will be explored by this call and the increase of the total cost must be consistent with the
    /// value of `peek_next` before this method was called.
    fn explore_next(&mut self) -> Option<Node>;

    fn current_node(&self) -> Node;

    fn explore(&mut self) -> Vec<Node> {
        let mut order = vec![self.current_node()];
        while let Some(node) = self.explore_next() {
            order.push(node);
        }
        order
    }
}

#[derive(Clone, Debug)]
pub struct PhantomAlg;

impl<'a, G> DetAlgorithm<'a, G> for PhantomAlg
where
    G: Graph<'a>,
{
    fn peek_next(&mut self) -> Action {
        Action::Finished
    }

    fn explore_next(&mut self) -> Option<Node> {
        None
    }

    fn current_node(&self) -> Node {
        Node::new(0)
    }
}
