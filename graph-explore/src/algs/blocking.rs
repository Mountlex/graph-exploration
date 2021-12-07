use crate::{
    cost::Cost,
    graph::{DetStartNode, Edge, EdgeSet, Graph, Node, Tree},
    state::State,
};

use super::{Action, DetAlgorithm, Move};

pub fn blocking<'a, G>(graph: &'a G) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let mut alg = Blocking::init(graph, start_node, 2.0, BlockingType::Default);
    alg.explore();
    alg.total_cost()
}

pub fn pblocking<'a, G>(graph: &'a G, tree: &'a Tree) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let mut alg = Blocking::init(graph, start_node, 0.0, BlockingType::Follow(tree));
    alg.explore();
    alg.total_cost()
}

#[derive(Debug, Clone)]
pub struct Blocking<'a, G>
where
    G: Graph<'a>,
{
    state: State<'a, G>,
    delta: f32,
    /// stack of: explored node v + set of edges which where blocked by an boundary edge to v.
    backtrack: Vec<(Node, EdgeSet)>,
    blocking_type: BlockingType<'a>,
    next_action: Option<Action>,
}

#[derive(Debug, Clone)]
pub enum BlockingType<'a> {
    Default,
    Follow(&'a Tree),
}

impl<'a, G> Blocking<'a, G>
where
    G: Graph<'a>,
{
    pub fn init(
        graph: &'a G,
        start_node: Node,
        delta: f32,
        blocking_type: BlockingType<'a>,
    ) -> Self {
        let state = State::cached(graph, start_node);

        Blocking {
            state,
            delta,
            backtrack: vec![(start_node, EdgeSet::default())],
            blocking_type,
            next_action: None,
        }
    }

    pub fn total_cost(&self) -> Cost {
        self.state.total_cost()
    }

    /// Returns whether `edge` is blocked by some other edge in `boundary.
    fn is_blocked_by(&self, edge: &Edge, boundary: &[Edge]) -> bool {
        for e in boundary {
            if self.is_blocked_by_edge(edge, e) {
                return true;
            }
        }

        false
    }

    /// Returns whether `edge` is blocked by `e_prime`.
    fn is_blocked_by_edge(&self, edge: &Edge, e_prime: &Edge) -> bool {
        let cost_for_e_prime = self.state.sp_between(e_prime.sink(), edge.source());
        e_prime.cost() < edge.cost() && cost_for_e_prime < edge.cost() * (1.0 + self.delta)
    }

    fn compute_next(&mut self) -> Action {
        let mut next: Option<Node> = None;

        // In the first iteration, y == current.
        while let Some((y, blocked_by_y)) = self.backtrack.last() {
            // try to find neighboring unblocked edge of y
            let neighbor_edges = self.state.unexplored_neighbor_edges_of(*y);
            let mut b_edges = self.filtered_boundary();
            for edge in neighbor_edges.clone().to_sink_sorted_vec() {
                if !self.is_blocked_by(&edge, &b_edges) {
                    next = Some(edge.sink());
                    break;
                }
            }

            if next.is_none() {
                // find other unblocked boundary edge which was previously blocked by some boundary edge incident to y.
                b_edges.sort_by_key(|e| e.sink());
                for edge in &b_edges {
                    if edge.sink() != *y
                        && !self.is_blocked_by(&edge, &b_edges)
                        && blocked_by_y.contains(&edge)
                    {
                        next = Some(edge.sink());
                        break;
                    }
                }
            }

            // Return next node if one has found in this iterations
            if let Some(v) = next {
                return Action::Move(Move {
                    current: self.state.current(),
                    cost: self.state.cost_sp_to(v),
                    next: v,
                });
            }
            // Here: next == None

            // Backtrack
            self.backtrack.pop();
        }

        Action::Finished
    }

    fn filtered_boundary(&self) -> Vec<Edge> {
        return self
            .state
            .boundary_edges()
            .into_iter()
            .filter(|e| match self.blocking_type {
                BlockingType::Default => true,
                BlockingType::Follow(t) => t.contains_edge(e.source(), e.sink()),
            })
            .collect();
    }
}

impl<'a, G> DetAlgorithm<'a, G> for Blocking<'a, G>
where
    G: Graph<'a> + DetStartNode<'a>,
{
    fn current_node(&self) -> Node {
        self.state.current()
    }

    fn peek_next(&mut self) -> Action {
        if let Some(ref action) = self.next_action {
            action.clone()
        } else {
            self.next_action = Some(self.compute_next());
            self.next_action.clone().unwrap()
        }
    }

    fn explore_next(&mut self) -> Option<Node> {
        // performs backtracking
        if let Action::Move(next_move) = self.peek_next() {
            let next = next_move.next;

            let boundary = self.filtered_boundary();

            // Compute edges which are currently blocked by at least one boundary edge to the next node.
            let mut blocked_by_boundary = EdgeSet::default();
            let boundary_to_next: EdgeSet = boundary
                .iter()
                .filter(|e| e.sink() == next)
                .copied()
                .collect();
            for edge in &boundary {
                if !boundary_to_next.contains(&edge) {
                    'inner: for b_edge in &boundary_to_next {
                        if self.is_blocked_by_edge(&edge, &b_edge) {
                            blocked_by_boundary.insert(*edge);
                            break 'inner;
                        }
                    }
                }
            }

            self.state.move_sp_to(next);

            // push new current node + blocked edges
            self.backtrack.push((next, blocked_by_boundary));

            self.next_action = None;
            Some(next)
        } else {
            self.state.back_to_start();
            None
        }
    }
}

#[cfg(test)]
mod test_blocking {
    use crate::graph::AdjListGraph;

    use super::*;

    ///   7 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|     |1|    
    ///            4 --1-- 5 --6-- 6
    fn get_graph() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 7.into(), 2.into());
        graph.add_edge(1.into(), 4.into(), 2.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        graph
    }

    ///   7 --2 -- 1 --5-- 2 --1-- 3
    ///           |2|     |1|    
    ///            4 --1-- 5 --6-- 6
    #[test]
    fn test_blocking() {
        let graph = get_graph();
        let mut alg = Blocking::init(&graph, 1.into(), 2.0, BlockingType::Default);

        assert!(alg.is_blocked_by(
            &Edge::new(1.into(), 2.into(), 5.into()),
            &alg.state.boundary_edges().collect::<Vec<Edge>>()
        ));
        assert!(alg.peek_next().is_move());
        assert_eq!(alg.peek_next().next_move().unwrap().next, 4.into());
        assert_eq!(alg.peek_next().next_move().unwrap().next, 4.into());
        assert_eq!(alg.peek_next().next_move().unwrap().next, 4.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 4.into());
        assert_eq!(alg.total_cost(), 2.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 5.into());
        assert_eq!(alg.total_cost(), 3.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 2.into());
        assert_eq!(alg.total_cost(), 4.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 3.into());
        assert_eq!(alg.total_cost(), 5.into());

        assert!(alg.is_blocked_by(
            &Edge::new(5.into(), 6.into(), 6.into()),
            &alg.state.boundary_edges().collect::<Vec<Edge>>()
        ));
        alg.explore_next();
        assert_eq!(alg.state.current(), 7.into());
        assert_eq!(alg.total_cost(), 12.into());

        assert!(!alg.is_blocked_by(
            &Edge::new(5.into(), 6.into(), 6.into()),
            &alg.state.boundary_edges().collect::<Vec<Edge>>()
        ));
        alg.explore_next();

        assert_eq!(alg.state.current(), 6.into());
        assert_eq!(alg.total_cost(), 23.into());

        assert!(alg.peek_next().is_finished());
        assert!(alg.state.graph_explored());
    }
}
