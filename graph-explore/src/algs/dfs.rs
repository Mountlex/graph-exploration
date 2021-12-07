use crate::{
    cost::Cost,
    graph::{DetStartNode, Edge, EdgeCond, Graph, Node, Tree},
    state::State,
};

use super::{Action, DetAlgorithm, Move};

pub fn dfs<'a, G>(graph: &'a G) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let mut alg = DepthFirstSearch::init(graph, start_node, DFSType::default());
    alg.explore();
    alg.total_cost()
}

pub fn pdfs<'a, G>(graph: &'a G, mst: &'a Tree) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let mut alg = DepthFirstSearch::init(graph, start_node, DFSType::follow_tree(mst));
    alg.explore();
    alg.total_cost()
}

/// Implementation of the Depth First Search (DFS) exploration algorithm.
///
/// **Note**: This implementation is deterministic and explores the neighbors of a node in order of edge costs, breaking ties using node indices.
#[derive(Clone, Debug)]
pub struct DepthFirstSearch<'a, G>
where
    G: Graph<'a>,
{
    state: State<'a, G>,
    stack: Vec<Node>,
    dfs_type: DFSType<'a>,
    next_action: Option<Action>,
}

#[derive(Clone, Debug)]
pub enum DFSType<'a> {
    Default,
    EdgeCond(Box<dyn EdgeCond>),
    Follow(&'a Tree),
}

impl<'a> DFSType<'a> {
    pub fn default() -> Self {
        DFSType::Default
    }

    pub fn with_edge_cond(edge_cond: Box<dyn EdgeCond>) -> Self {
        DFSType::EdgeCond(edge_cond)
    }

    pub fn follow_tree(tree: &'a Tree) -> Self {
        DFSType::Follow(tree)
    }
}

impl<'a, G> DepthFirstSearch<'a, G>
where
    G: Graph<'a>,
{
    pub fn init(graph: &'a G, start_node: Node, dfs_type: DFSType<'a>) -> Self {
        let state = State::<'a>::cached(graph, start_node);

        let mut dfs = DepthFirstSearch {
            state,
            stack: vec![],
            dfs_type,
            next_action: None,
        };
        dfs.add_to_stack();
        dfs
    }

    pub fn total_cost(&self) -> Cost {
        self.state.total_cost()
    }

    fn add_to_stack(&mut self) {
        let mut edges: Vec<Edge> = self.state.unexplored_neighbor_edges().into_iter().collect();
        edges.sort_by(|e1, e2| e1.cost().cmp(&e2.cost()).then(e1.sink().cmp(&e2.sink())));
        edges.reverse();

        let mut unexplored_valid_neighbors: Vec<Node> = edges
            .into_iter()
            .filter(|e| match &self.dfs_type {
                DFSType::Default => true,
                DFSType::EdgeCond(cond) => (*cond).check(e),
                DFSType::Follow(tree) => tree.contains_edge(e.source(), e.sink()),
            })
            .map(|e| e.sink())
            .collect();

        self.stack.append(&mut unexplored_valid_neighbors)
    }

    fn compute_next(&mut self) -> Action {
        let next_explored: Node;
        loop {
            if let Some(&top) = self.stack.last() {
                if self.state.is_explored(top) {
                    self.stack.pop();
                } else {
                    next_explored = top;
                    break;
                }
            } else {
                return Action::Finished; // There is no unexplored node in the stack.
            }
        }
        let cost = self.state.cost_sp_to(next_explored);
        Action::Move(Move {
            current: self.state.current(),
            cost,
            next: next_explored,
        })
    }
}

impl<'a, G> DetAlgorithm<'a, G> for DepthFirstSearch<'a, G>
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
        if let Action::Move(next_move) = self.peek_next() {
            //assert!(*self.stack.last().unwrap() == next_move.next);
            self.stack.pop();

            // move a shortest path to the next vertex
            self.state.move_sp_to(next_move.next);

            // append all unexplored neighbors at the new current vertex in reversed order of their index to the stack
            self.add_to_stack();

            self.next_action = None;
            Some(next_move.next)
        } else {
            self.state.back_to_start();

            None
        }
    }
}

#[cfg(test)]
mod test_dfs {
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
    fn test_dfs() {
        let graph = get_graph();
        let mut dfs = DepthFirstSearch::init(&graph, Node::new(1), DFSType::default());

        assert!(dfs.peek_next().is_move());

        // Test determinism
        assert_eq!(dfs.peek_next().next_move().unwrap().next, 4.into());
        assert_eq!(dfs.peek_next().next_move().unwrap().next, 4.into());
        assert_eq!(dfs.peek_next().next_move().unwrap().next, 4.into());

        let next_action = dfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.cost, 2.into());
        dfs.explore_next();
        assert_eq!(dfs.state.current(), 4.into());
        assert_eq!(dfs.total_cost(), 2.into());

        let next_action = dfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 5.into());
        assert_eq!(next_action.cost, 1.into());
        dfs.explore_next();
        assert_eq!(dfs.state.current(), 5.into());
        assert_eq!(dfs.total_cost(), 3.into());

        let next_action = dfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 2.into());
        assert_eq!(next_action.cost, 1.into());
        dfs.explore_next();

        let next_action = dfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 3.into());
        assert_eq!(next_action.cost, 1.into());
        dfs.explore_next();

        let next_action = dfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 6.into());
        assert_eq!(next_action.cost, 8.into());
        dfs.explore_next();

        let next_action = dfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 7.into());
        assert_eq!(next_action.cost, 11.into());
        dfs.explore_next();

        assert!(dfs.state.graph_explored());
        assert!(dfs.peek_next().is_finished());
    }
}
