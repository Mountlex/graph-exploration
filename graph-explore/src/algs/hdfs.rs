use crate::{
    cost::Cost,
    graph::{
        AdjListGraph, Cut, DetStartNode, EdgeSet, ExpRoundable, Graph, GraphView, Node, Nodes,
    },
    mst::prims_tree,
    state::State,
};

use super::{Action, DetAlgorithm, Move};

pub fn hdfs(graph: &AdjListGraph) -> Cost {
    let start_node = graph.start_node();
    let mut alg = HierarchicalDFS::init(graph, start_node);
    alg.explore();
    alg.total_cost()
}

pub fn hdfs_rounding(graph: &AdjListGraph) -> Cost {
    let start_node = graph.start_node();
    let rounded_graph = graph.to_exp_rounded();
    let mut alg = HierarchicalDFS::init_rounded(&rounded_graph, &graph, start_node);
    alg.explore();
    let cost = alg.total_cost();
    drop(graph);
    drop(rounded_graph);
    cost
}

#[derive(Clone, Debug)]
pub struct HierarchicalDFS<'a> {
    decision_state: State<'a, AdjListGraph>,
    move_state: Option<State<'a, AdjListGraph>>,
    stack: Vec<(Node, Cost, Option<Vec<Node>>)>,
    next_action: Option<Action>,
}

impl<'a> HierarchicalDFS<'a> {
    pub fn init(graph: &'a AdjListGraph, start_node: Node) -> Self {
        HierarchicalDFS {
            decision_state: State::cached(graph, start_node),
            move_state: None,
            stack: vec![(start_node, Cost::max(), None)],
            next_action: None,
        }
    }

    pub fn init_rounded(
        rounded_graph: &'a AdjListGraph,
        base_graph: &'a AdjListGraph,
        start_node: Node,
    ) -> Self {
        HierarchicalDFS {
            decision_state: State::cached(rounded_graph, start_node),
            move_state: Some(State::cached(base_graph, start_node)),
            stack: vec![(start_node, Cost::max(), None)],
            next_action: None,
        }
    }

    pub fn total_cost(&self) -> Cost {
        if let Some(move_state) = &self.move_state {
            move_state.total_cost()
        } else {
            self.decision_state.total_cost()
        }
    }

    fn compute_next(&mut self) -> Action {
        'call_stack: while let Some((u, w, mst_order)) = self.stack.last_mut() {
            let comp_u_excl = GraphView::with_upper_bound_strict(self.decision_state.graph, *u, *w);

            if mst_order.is_none() {
                // Line 1: search for a smaller weight and add a call to the stack if one exists
                let comp_u = GraphView::with_upper_bound(self.decision_state.graph, *u, *w);
                if let Some(w_prime) = comp_u
                    .cut(&self.decision_state.explored)
                    .map(|e| e.cost())
                    .min()
                {
                    if w_prime < *w {
                        let u_copy = *u; // this allocation is for compiler reasons, otherwise we have https://github.com/rust-lang/rust/issues/59159

                        // Line 2: if a smaller weight was found, call hDFS on this weight.
                        self.stack.push((u_copy, w_prime, None));
                        continue 'call_stack;
                    }
                }

                // Line 4: compute mst order and save it.
                let mst = prims_tree(&comp_u_excl).0; // TODO: Replace by some more efficient tree data structure.
                assert!(mst.contains_node(*u));
                *mst_order = Some(mst.nodes().collect::<Vec<Node>>());
            }

            // Line 5: search for boundary edges of weight w and explore them in the MST order.
            let w_boundary: EdgeSet = self
                .decision_state
                .boundary_edges()
                .filter(|e| comp_u_excl.contains_node(e.source()) && e.cost() == *w)
                .collect();

            if !w_boundary.is_empty() {
                let sorted_boundary = w_boundary.to_sink_sorted_vec(); // for determinism
                for node in mst_order.as_ref().unwrap() {
                    for edge in &sorted_boundary {
                        // Line 6: get next boundary edge in mst order.
                        if edge.source() == *node {
                            let next = edge.sink();
                            let w_copy = *w; // this allocation is for compiler reasons, see above.

                            // Line 8: call hDFS on next.
                            self.stack.push((next, w_copy, None));

                            // Line 7: return action to traverse a shortest path to next.
                            let cost = if let Some(move_state) = &self.move_state {
                                move_state.cost_sp_to(next)
                            } else {
                                self.decision_state.cost_sp_to(next)
                            };
                            return Action::Move(Move {
                                next,
                                current: self.decision_state.current(),
                                cost,
                            });
                        }
                    }
                }
            }

            // line 9: no more w_boundary edges to explore -> backtrack
            self.stack.pop();
        }

        Action::Finished
    }
}

impl<'a> DetAlgorithm<'a, AdjListGraph> for HierarchicalDFS<'a> {
    fn current_node(&self) -> Node {
        self.decision_state.current()
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
        if let Action::Move(action) = self.peek_next() {
            let next = action.next;
            self.decision_state.move_sp_to(next);
            if let Some(move_state) = &mut self.move_state {
                move_state.move_sp_to(next);
            }

            self.next_action = None;
            Some(action.next)
        } else {
            self.decision_state.back_to_start();

            if let Some(move_state) = &mut self.move_state {
                move_state.back_to_start();
            }
            None
        }
    }
}

#[cfg(test)]
mod test_hdfs {
    use crate::graph::AdjListGraph;

    use super::*;

    /// 6 --8-- 4 --2-- 1 --1-- 2 --1-- 3 --4-- 5 --16-- 7
    fn get_graph1() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(1.into(), 4.into(), 2.into());
        graph.add_edge(3.into(), 5.into(), 4.into());
        graph.add_edge(4.into(), 6.into(), 8.into());
        graph.add_edge(5.into(), 7.into(), 16.into());

        graph
    }

    #[test]
    fn test_hdfs_graph1() {
        let graph = get_graph1();
        let mut hdfs = HierarchicalDFS::init(&graph, Node::new(1));

        assert!(hdfs.peek_next().is_move());

        // Test determinism
        assert_eq!(hdfs.peek_next().next_move().unwrap().next, 2.into());
        assert_eq!(hdfs.peek_next().next_move().unwrap().next, 2.into());
        assert_eq!(hdfs.peek_next().next_move().unwrap().next, 2.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.current, 1.into());
        assert_eq!(next_action.cost, 1.into());
        hdfs.explore_next();
        assert_eq!(hdfs.decision_state.current(), 2.into());
        assert_eq!(hdfs.total_cost(), 1.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 3.into());
        assert_eq!(next_action.cost, 1.into());
        hdfs.explore_next();
        assert_eq!(hdfs.decision_state.current(), 3.into());
        assert_eq!(hdfs.total_cost(), 2.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 4.into());
        assert_eq!(next_action.cost, 4.into());
        hdfs.explore_next();
        assert_eq!(hdfs.decision_state.current(), 4.into());
        assert_eq!(hdfs.total_cost(), 6.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 5.into());
        assert_eq!(next_action.cost, 8.into());
        hdfs.explore_next();
        assert_eq!(hdfs.decision_state.current(), 5.into());
        assert_eq!(hdfs.total_cost(), 14.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 6.into());
        assert_eq!(next_action.cost, 16.into());
        hdfs.explore_next();
        assert_eq!(hdfs.decision_state.current(), 6.into());
        assert_eq!(hdfs.total_cost(), 30.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 7.into());
        assert_eq!(next_action.cost, 32.into());
        hdfs.explore_next();
        assert_eq!(hdfs.decision_state.current(), 7.into());
        assert_eq!(hdfs.total_cost(), 62.into());
    }

    ///    7 --2-- 1 --5-- 2 --1-- 3
    ///           |2|     |1|    
    ///            4 --1-- 5 --6-- 6
    fn get_graph2() -> AdjListGraph {
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

    #[test]
    fn test_hdfs_graph2() {
        let graph = get_graph2();
        let mut hdfs = HierarchicalDFS::init(&graph, Node::new(1));

        assert!(hdfs.peek_next().is_move());

        // Test determinism
        assert_eq!(hdfs.peek_next().next_move().unwrap().next, 4.into());
        assert_eq!(hdfs.peek_next().next_move().unwrap().next, 4.into());
        assert_eq!(hdfs.peek_next().next_move().unwrap().next, 4.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.current, 1.into());
        assert_eq!(next_action.cost, 2.into());
        hdfs.explore_next();
        assert_eq!(hdfs.decision_state.current(), 4.into());
        assert_eq!(hdfs.total_cost(), 2.into());

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 5.into());
        assert_eq!(next_action.cost, 1.into());
        hdfs.explore_next();

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 2.into());
        assert_eq!(next_action.cost, 1.into());
        hdfs.explore_next();

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 3.into());
        assert_eq!(next_action.cost, 1.into());
        hdfs.explore_next();

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 7.into());
        assert_eq!(next_action.cost, 7.into());
        hdfs.explore_next();

        let next_action = hdfs.peek_next().next_move().unwrap();
        assert_eq!(next_action.next, 6.into());
        assert_eq!(next_action.cost, 11.into());
        hdfs.explore_next();
    }
}
