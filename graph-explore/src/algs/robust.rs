use crate::{
    cost::Cost,
    graph::{AdjListGraph, DetStartNode, ExpRoundable, Graph, NeighborTour, Node},
    state::State,
};
use std::marker::PhantomData;

use super::{
    Action, Blocking, BlockingType, DetAlgorithm, HierarchicalDFS, Move, PhantomAlg, Salesperson,
};

pub fn robust<'a, G, A, H>(graph: &'a G, lambda: f64, b_box: A) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
    H: Graph<'a>,
    A: DetAlgorithm<'a, G> + Clone,
{
    let start_node = graph.start_node();
    let mut alg = Robust::init(graph, start_node, lambda, 1.0, b_box);
    alg.explore();
    alg.total_cost()
}

pub fn robust_ptour<'a, G>(graph: &'a G, lambda: f64, pred: &'a NeighborTour) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let p_tour = Salesperson::<'a, G>::init(graph, start_node, pred);
    let mut alg = Robust::init(graph, start_node, lambda, 1.0, p_tour);
    alg.explore();
    alg.total_cost()
}

pub fn theoretic_robust_ptour<'a, G>(graph: &'a G, lambda: f64, pred: &'a NeighborTour) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let p_tour = Salesperson::<'a, G>::init(graph, start_node, pred);
    let mut alg = Robust::init(graph, start_node, lambda, 0.0, p_tour);
    alg.explore();
    alg.total_cost()
}

pub fn robust_hdfs(graph: &AdjListGraph, lambda: f64) -> Cost {
    let start_node = graph.start_node();
    let p_tour = HierarchicalDFS::init(graph, start_node);
    let mut alg = Robust::init(graph, start_node, lambda, 1.0, p_tour);
    alg.explore();
    alg.total_cost()
}

pub fn robust_hdfs_rounded(graph: &AdjListGraph, lambda: f64) -> Cost {
    let start_node = graph.start_node();
    let rounded = graph.to_exp_rounded();
    let p_tour = HierarchicalDFS::init_rounded(&rounded, graph, start_node);
    let mut alg = Robust::init(graph, start_node, lambda, 1.0, p_tour);
    alg.explore();
    alg.total_cost()
}

pub fn robust_blocking<'a>(graph: &'a AdjListGraph, lambda: f64, delta: f32) -> Cost {
    let start_node = graph.start_node();
    let blocking = Blocking::<'a>::init(graph, start_node, delta, BlockingType::Default);
    let mut alg = Robust::init(graph, start_node, lambda, 1.0, blocking);
    alg.explore();
    alg.total_cost()
}

pub fn nearest_neighbor<'a, G>(graph: &'a G) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let mut alg: Robust<'a, G, PhantomAlg, G> =
        Robust::init(graph, start_node, f64::MAX, 0.0, PhantomAlg);
    alg.explore();
    alg.total_cost()
}

#[derive(Clone, Debug)]
pub struct Robust<'a, G, A, H>
where
    G: Graph<'a>,
{
    state: State<'a, G>,
    lambda: f64,
    gamma: f64,
    blackbox: A,
    _blackbox_graph_type: PhantomData<H>,
    current_config: Configuration,
    next_action: Option<Action>,
}

#[derive(Clone, Debug)]
struct Configuration {
    blackbox_target: Option<Node>,
    nn_cost: f64,
    nn_threshold: f64,
    nn_total_cost: f64,
    blackbox_cost: f64,
}

impl<'a, G, A, H> Robust<'a, G, A, H>
where
    G: Graph<'a>,
    H: Graph<'a>,
    A: DetAlgorithm<'a, H>,
{
    pub fn init(graph: &'a G, start_node: Node, lambda: f64, gamma: f64, blackbox: A) -> Self {
        // For now gamma should be either 0.0 or 1.0:
        assert!(gamma == 0.0 || gamma == 1.0);

        let state = State::cached(graph, start_node);
        let current_config = Configuration {
            blackbox_target: None,
            nn_cost: 0.0,
            nn_threshold: 0.0,
            nn_total_cost: 0.0,
            blackbox_cost: 0.0,
        };

        Robust {
            state,
            lambda,
            gamma,
            blackbox,
            _blackbox_graph_type: PhantomData,
            current_config,
            next_action: None,
        }
    }

    pub fn total_cost(&self) -> Cost {
        self.state.total_cost()
    }

    fn compute_next(&mut self) -> Action {
        if !self.state.graph_explored() {
            let nn_target = self.find_nn().expect(&format!(
                "The nearest neighbor should not be None at this point (explored={}, n={}).",
                self.state.explored.len(),
                self.state.graph.n()
            ));

            // If the current blackbox target is None, compute the target, and set the budget.
            if self.current_config.blackbox_target.is_none() {
                if let Some(node) = self.next_blackbox_target() {
                    self.current_config.blackbox_target = Some(node);
                    //self.current_config.nn_cost = 0.0;
                    //self.current_config.nn_threshold =
                    //    self.state.cost_sp_to(node).as_float();
                }
                // This else-part is only relevant for pure NN.
                else {
                    // Update the configuration and return the action:
                    let add_cost = self.state.cost_sp_to(nn_target);
                    self.current_config.nn_cost = self.current_config.nn_cost + add_cost.as_float();
                    self.current_config.nn_total_cost =
                        self.current_config.nn_total_cost + add_cost.as_float();
                    return Action::Move(Move {
                        next: nn_target,
                        cost: add_cost,
                        current: self.state.current(),
                    });
                }
            }

            let b_target = self
                .current_config
                .blackbox_target
                .expect("The next blackbox vertex should not be None at this point.");

            // Execute the blackbox until the next move would surpass cost (1/lambda) * nn
            let b_target_cost = self.state.cost_sp_to(b_target).as_float();
            if self.current_config.blackbox_cost + b_target_cost
                <= (1.0 / self.lambda) * self.gamma * self.current_config.nn_total_cost
            {
                self.current_config.blackbox_cost =
                    self.current_config.blackbox_cost + b_target_cost;
                // self.blackbox.explore_next();
                // self.current_config.blackbox_target = self.next_blackbox_target();
                self.current_config.blackbox_target = None;
                return Action::Move(Move {
                    next: b_target,
                    cost: self.state.cost_sp_to(b_target),
                    current: self.state.current(),
                });
            }

            // If we reach this point, we should set the nn_threshold
            self.current_config.nn_threshold = self.state.cost_sp_to(b_target).as_float();

            // Cost of the next NN move:
            let add_cost = self.state.cost_sp_to(nn_target);

            // If the current nn search budget is not empty yet and the nearest neighbor does not match the blackbox target, return the next nn action:
            if (self.current_config.nn_cost
                < (self.current_config.nn_threshold
                    + self.current_config.blackbox_cost * self.gamma)
                    * self.lambda
                && b_target.id() != nn_target.id())
                || (self.current_config.nn_cost + add_cost.as_float()
                    < self.lambda * self.gamma * self.current_config.nn_total_cost
                    && b_target.id() == nn_target.id())
            {
                // Update the configuration and return the action:
                self.current_config.nn_cost = self.current_config.nn_cost + add_cost.as_float();
                self.current_config.nn_total_cost =
                    self.current_config.nn_total_cost + add_cost.as_float();

                // If necessary, compute a new blackbox target:
                if b_target.id() == nn_target.id() {
                    //self.blackbox.explore_next();
                    self.current_config.blackbox_target = self.next_blackbox_target();
                }

                return Action::Move(Move {
                    next: nn_target,
                    cost: add_cost,
                    current: self.state.current(),
                });
            // Otherwise, reset the configuration and return the blackbox action:
            } else {
                // Reset the configuration:
                self.current_config.blackbox_target = None;
                self.current_config.nn_cost = 0.0;
                self.current_config.nn_threshold = 0.0;
                self.current_config.blackbox_cost = 0.0;

                let add_cost = self.state.cost_sp_to(b_target);
                return Action::Move(Move {
                    next: b_target,
                    cost: add_cost,
                    current: self.state.current(),
                });
            }
        }
        Action::Finished
    }

    // Returns the current nearest neighbor:
    fn find_nn(&self) -> Option<Node> {
        let boundary_nodes = self.state.boundary_nodes().clone();
        let boundary_vc = boundary_nodes.to_sorted_vec();

        let mut i = 0;
        let mut min_cost = Cost::max();
        let mut nn = None;

        while i < boundary_vc.len() {
            let vertex = boundary_vc[i];
            let c = self.state.cost_sp_to(vertex);
            if c < min_cost {
                min_cost = c;
                nn = Some(vertex);
            }
            i = i + 1;
        }

        nn
    }

    // Returns the current target of the blackbox:
    fn next_blackbox_target(&mut self) -> Option<Node> {
        if let Action::Move(action) = self.blackbox.peek_next() {
            // Simulate algorithmic moves of the blackbox to already explored vertices:
            let mut next_vertex = action.next;
            while self.state.explored.contains(&next_vertex) {
                self.blackbox.explore_next();
                match self.blackbox.peek_next() {
                    Action::Finished => return None,
                    Action::Move(m) => {
                        next_vertex = m.next;
                    }
                }
            }
            return Some(next_vertex);
        }
        None
    }
}

impl<'a, G, A, H> DetAlgorithm<'a, G> for Robust<'a, G, A, H>
where
    G: Graph<'a> + DetStartNode<'a>,
    H: Graph<'a>,
    A: DetAlgorithm<'a, H>,
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
        if let Action::Move(next_action) = self.peek_next() {
            // move a shortest path to the next vertex
            self.state.move_sp_to(next_action.next);

            self.next_action = None;
            Some(next_action.next)
        } else {
            self.state.back_to_start();
            None
        }
    }
}

#[cfg(test)]
mod test_robust {
    use super::*;
    use crate::algs::{DFSType, DepthFirstSearch};
    use crate::graph::AdjListGraph;

    fn get_graph() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 3.into());
        graph.add_edge(1.into(), 3.into(), 1.into());

        graph.add_edge(3.into(), 4.into(), 3.into());
        graph.add_edge(4.into(), 9.into(), 1.into());

        graph.add_edge(3.into(), 6.into(), 1.into());
        graph.add_edge(6.into(), 7.into(), 1.into());

        graph.add_edge(7.into(), 8.into(), 4.into());
        graph.add_edge(8.into(), 9.into(), 6.into());

        graph
    }

    #[test]
    fn robust_t1() {
        let l = 0.0;
        let graph = get_graph();
        let dfs = DepthFirstSearch::init(&graph, Node::new(1), DFSType::default());
        let mut alg = Robust::init(&graph, 1.into(), l, 0.0, dfs);

        assert!(alg.peek_next().is_move());
        assert_eq!(alg.peek_next().next_move().unwrap().next, 3.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 3.into());
        assert_eq!(alg.state.total_cost(), 1.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 6.into());
        assert_eq!(alg.state.total_cost(), 2.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 7.into());
        assert_eq!(alg.state.total_cost(), 3.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 8.into());
        assert_eq!(alg.state.total_cost(), 7.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 9.into());
        assert_eq!(alg.state.total_cost(), 13.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 4.into());
        assert_eq!(alg.state.total_cost(), 14.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 2.into());
        assert_eq!(alg.state.total_cost(), 21.into());
    }

    #[test]
    fn robust_t5() {
        let l = 20.0;
        let graph = get_graph();
        let dfs = DepthFirstSearch::init(&graph, Node::new(1), DFSType::default());
        let mut alg = Robust::init(&graph, 1.into(), l, 0.0, dfs);

        assert!(alg.peek_next().is_move());
        assert_eq!(alg.peek_next().next_move().unwrap().next, 3.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 3.into());
        assert_eq!(alg.state.total_cost(), 1.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 6.into());
        assert_eq!(alg.state.total_cost(), 2.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 7.into());
        assert_eq!(alg.state.total_cost(), 3.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 8.into());
        assert_eq!(alg.state.total_cost(), 7.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 9.into());
        assert_eq!(alg.state.total_cost(), 13.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 4.into());
        assert_eq!(alg.state.total_cost(), 14.into());

        alg.explore_next();
        assert_eq!(alg.state.current(), 2.into());
        assert_eq!(alg.state.total_cost(), 21.into());
    }
}
