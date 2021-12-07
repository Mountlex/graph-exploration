use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::{Display, Formatter, Result},
};

use binary_heap_plus::BinaryHeap;
use compare::Compare;
use ndarray::Array2;

use crate::{cost::Cost, graph::NodeIndex};
use crate::{
    dijkstra::dijkstra,
    graph::{Cut, DetStartNode, Edge, EdgeSet, Graph, Node, NodeSet},
};

#[derive(Clone, Debug, PartialEq, Eq)]
enum PathCost {
    Unreachable,
    Path(Cost),
}

impl Display for PathCost {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            PathCost::Path(cost) => write!(f, "{}", cost),
            PathCost::Unreachable => write!(f, "∞"),
        }
    }
}

impl PathCost {
    fn is_unreachable(&self) -> bool {
        match self {
            Self::Unreachable => true,
            _ => false,
        }
    }

    fn cost(&self) -> Option<&Cost> {
        match self {
            Self::Path(cost) => Some(cost),
            _ => None,
        }
    }
}

/// Represents the state of an exploration algorithm on a graph.
#[derive(Debug, Clone)]
pub struct State<'a, G>
where
    G: Graph<'a>,
{
    sp: BoundaryPathCache<'a, G>,
    pub graph: &'a G,
    pub explored: NodeSet,
    unexplored_boundary: NodeSet,
    explored_boundary: NodeSet,
    current: Node,
    total_cost: Cost,
}

impl<'a, G> State<'a, G>
where
    G: Graph<'a>,
{
    pub fn cached(graph: &'a G, start_node: Node) -> Self {
        let explored = NodeSet::singleton(start_node);
        let unexplored_boundary = graph.cut(&explored).map(|e| e.sink()).collect();
        let explored_boundary = NodeSet::singleton(start_node);

        State {
            sp: BoundaryPathCache::all_pairs_cache(graph, start_node),
            graph,
            explored,
            explored_boundary,
            unexplored_boundary,
            current: start_node,
            total_cost: Cost::new(0),
        }
    }

    #[allow(dead_code)]
    pub fn lazy(graph: &'a G, start_node: Node) -> Self {
        let explored = NodeSet::singleton(start_node);
        let unexplored_boundary = graph.cut(&explored).map(|e| e.sink()).collect();
        let explored_boundary = NodeSet::singleton(start_node);

        State {
            sp: BoundaryPathCache::lazy_eval(graph, start_node),
            graph,
            explored,
            explored_boundary,
            unexplored_boundary,
            current: start_node,
            total_cost: Cost::new(0),
        }
    }

    pub fn graph_explored(&self) -> bool {
        self.explored == self.graph.nodes().into_iter().collect()
    }

    pub fn boundary_edges(&self) -> <G as Cut<'a>>::CutIter {
        self.graph.cut(&self.explored)
    }

    pub fn boundary_nodes(&self) -> &NodeSet {
        &self.unexplored_boundary
    }

    pub fn visible_nodes(&self) -> NodeSet {
        self.explored
            .union(&self.boundary_nodes())
            .copied()
            .collect()
    }

    pub fn current(&self) -> Node {
        self.current
    }

    pub fn total_cost(&self) -> Cost {
        self.total_cost
    }

    pub fn is_explored(&self, node: Node) -> bool {
        self.explored.contains(&node)
    }

    pub fn unexplored_neighbor_edges_of(&self, node: Node) -> EdgeSet {
        let mut edges = EdgeSet::default();
        for edge in self.graph.adjacent(node) {
            if !self.explored.contains(&edge.sink()) {
                edges.insert(edge.clone());
            }
        }
        edges
    }

    pub fn unexplored_neighbor_edges(&self) -> EdgeSet {
        self.unexplored_neighbor_edges_of(self.current)
    }

    pub fn sp_between(&self, node1: Node, node2: Node) -> Cost {
        debug_assert!(self
            .boundary_edges()
            .into_iter()
            .any(|e| e.sink() == node1 || e.source() == node1));
        debug_assert!(self
            .boundary_edges()
            .into_iter()
            .any(|e| e.sink() == node2 || e.source() == node2));
        self.sp.shortest_path(node1, node2)
    }

    pub fn cost_sp_to(&self, next: Node) -> Cost {
        debug_assert!(self
            .boundary_edges()
            .into_iter()
            .any(|e| e.sink() == next || e.source() == next));
        self.sp.shortest_path(self.current, next)
    }

    pub fn back_to_start(&mut self) {
        debug_assert!(self.graph_explored());
        let cost = dijkstra(self.graph, self.current, self.graph.start_node());
        self.total_cost += cost;
    }

    pub fn move_sp_to(&mut self, next: Node) {
        debug_assert!(!self.explored.contains(&next));
        debug_assert!(self.boundary_nodes().contains(&next));

        self.total_cost += self.sp.shortest_path(self.current, next);

        // 1
        if !self.explored.contains(&next) {
            self.sp
                .update_shortest_paths(&self.explored, &self.visible_nodes(), next);
        }

        // 2
        self.explored.insert(next);
        self.unexplored_boundary.remove(next);
        let mut unexplored_neighbor = false;
        for node in self.graph.neighbors(next) {
            if !self.explored.contains(&node) {
                unexplored_neighbor = true;
                self.unexplored_boundary.insert(node);
            } else {
                if self
                    .graph
                    .neighbors(node)
                    .all(|n| self.explored.contains(&n))
                {
                    self.explored_boundary.remove(node);
                }
            }
        }
        if unexplored_neighbor {
            self.explored_boundary.insert(next);
        }

        self.current = next;
    }
}

struct PrioComp(HashMap<Node, Cost>);

impl Compare<Node> for PrioComp {
    fn compare(&self, l: &Node, r: &Node) -> Ordering {
        self.0
            .get(r)
            .unwrap()
            .cmp(self.0.get(l).unwrap())
            .then(r.cmp(l))
    }
}

#[derive(Debug, Clone)]
enum BoundaryPathCache<'a, G> {
    AllPairsCache(ShortestPaths<'a, G>),
    LazyEval(&'a G, NodeSet, NodeSet),
}

impl<'a, G> BoundaryPathCache<'a, G>
where
    G: Graph<'a>,
{
    fn all_pairs_cache(graph: &'a G, start_node: Node) -> Self {
        Self::AllPairsCache(ShortestPaths::init(graph, start_node))
    }

    fn lazy_eval(graph: &'a G, start_node: Node) -> Self {
        let mut visible = graph.neighbors(start_node).collect::<NodeSet>();
        visible.insert(start_node);
        Self::LazyEval(graph, NodeSet::singleton(start_node), visible)
    }

    fn shortest_path(&self, n1: Node, n2: Node) -> Cost {
        match self {
            BoundaryPathCache::AllPairsCache(sp) => sp.shortest_path(n1, n2),
            BoundaryPathCache::LazyEval(graph, explored, visible) => {
                let boundary: NodeSet = visible.diff(explored).copied().collect();
                let mut costs = HashMap::<Node, Cost>::with_capacity(graph.n());

                let nodes_vec: Vec<Node> = visible.to_vec();
                for n in nodes_vec.iter() {
                    costs.insert(*n, Cost::max());
                }
                costs.insert(n1, 0.into());

                let mut heap = BinaryHeap::from_vec_cmp(nodes_vec, PrioComp(costs.clone()));
                while let Some(u) = heap.pop() {
                    if u == n2 {
                        break;
                    }
                    for edge in graph.adjacent(u) {
                        if visible.contains(&edge.sink())
                            && !(boundary.contains(&edge.source())
                                && boundary.contains(&edge.sink()))
                        {
                            let update = *costs.get(&u).unwrap() + edge.cost();
                            let dist_v = costs.get_mut(&edge.sink()).unwrap();
                            if update < *dist_v {
                                *dist_v = update;
                            }
                        }
                    }
                    heap.replace_cmp(PrioComp(costs.clone()));
                }
                *costs.get(&n2).unwrap()
            }
        }
    }

    fn update_shortest_paths(
        &mut self,
        unexplored_boundary: &NodeSet,
        explored_boundary: &NodeSet,
        v: Node,
    ) {
        match self {
            BoundaryPathCache::AllPairsCache(sp) => {
                sp.update_shortest_paths(unexplored_boundary, explored_boundary, v)
            }
            BoundaryPathCache::LazyEval(graph, explored, visible) => {
                explored.insert(v);
                *visible = visible
                    .union(&graph.neighbors(v).collect::<NodeSet>())
                    .copied()
                    .collect::<NodeSet>()
            }
        }
    }
}

#[derive(Debug, Clone)]
struct ShortestPaths<'a, G> {
    graph: &'a G,
    matrix: Array2<PathCost>,
    node_index: NodeIndex,
}

impl<'a, G> ShortestPaths<'a, G>
where
    G: Graph<'a>,
{
    fn init(graph: &'a G, start_node: Node) -> Self {
        let mut nodes = graph.nodes().collect::<Vec<Node>>();
        nodes.sort();
        let n = nodes.len();
        let mut d = Array2::from_elem((n, n), PathCost::Unreachable);
        for i in 0..n {
            d[[i, i]] = PathCost::Path(0.into());
        }

        let mut sp = ShortestPaths {
            graph,
            matrix: d,
            node_index: NodeIndex::init(&nodes),
        };
        let mut visible_nodes = vec![start_node];
        for edge in graph.adjacent(start_node) {
            visible_nodes.push(edge.sink());
            sp.set(edge.source(), edge.sink(), edge.cost());
        }

        sp.search_shortcuts(visible_nodes, start_node);
        sp
    }

    /// We assume that the explored nodes have not been updated yet!
    /// Essentially, this is the Bellman-Ford update subroutine that is executed each time a node is explored.
    fn update_shortest_paths(
        &mut self,
        unexplored_boundary: &NodeSet,
        explored_boundary: &NodeSet,
        v: Node,
    ) {
        let adj_of_v: Vec<Edge> = self.graph.adjacent(v).collect();

        let relevant_nodes: NodeSet = unexplored_boundary
            .union(explored_boundary)
            .copied()
            .collect();
        debug_assert!(relevant_nodes.contains(&v));

        //Update shortest paths from all relevant nodes to v
        for rel_node in &relevant_nodes {
            if v != rel_node {
                // v was boundary node before
                for e in &adj_of_v {
                    let n = e.sink();
                    if relevant_nodes.contains(&n) {
                        let n_to_v = e.cost();
                        let rel_to_n = self.shortest_path(rel_node, n);
                        let rel_to_v = self.shortest_path(rel_node, v);
                        if n_to_v + rel_to_n < rel_to_v {
                            self.set(rel_node, v, n_to_v + rel_to_n);
                        }
                    }
                }
            }
        }

        let mut revealed_nodes = NodeSet::empty(); // sink nodes of v which were not visible before.
                                                   // Update shortest paths to newly revealed neighbors of v. Shortest paths to them must run through v.
        for e in adj_of_v {
            let b = e.sink();
            if !relevant_nodes.contains(&b) {
                revealed_nodes.insert(b);
                for n in &relevant_nodes {
                    let cost_of_boundary = e.cost();
                    self.set(b, n, self.shortest_path(n, v) + cost_of_boundary);
                }
            }
        }

        self.search_shortcuts(relevant_nodes.union(&revealed_nodes).copied(), v);
    }

    fn get(&self, n1: Node, n2: Node) -> &PathCost {
        let i1 = self.node_index[&n1];
        let i2 = self.node_index[&n2];

        let x = i1.min(i2);
        let y = i1.max(i2);
        &self.matrix[[x, y]]
    }

    fn set(&mut self, n1: Node, n2: Node, cost: Cost) {
        let i1 = self.node_index[&n1];
        let i2 = self.node_index[&n2];

        let x = i1.min(i2);
        let y = i1.max(i2);
        self.matrix[[x, y]] = PathCost::Path(cost);
    }

    fn search_shortcuts<I>(&mut self, nodes: I, v: Node)
    where
        I: IntoIterator<Item = Node>,
    {
        // Check if shortest paths among all visible nodes improves when going through v
        for n1 in nodes {
            for n2 in self.graph.neighbors(v) {
                if n1 < n2 && n1 != v && n2 != v {
                    let n1_to_v = self.shortest_path(n1, v);
                    let n2_to_v = self.shortest_path(n2, v);
                    let n1_to_n2 = self.get(n1, n2);

                    if n1_to_n2.is_unreachable() || n2_to_v + n1_to_v < *n1_to_n2.cost().unwrap() {
                        self.set(n1, n2, n2_to_v + n1_to_v)
                    }
                }
            }
        }
    }

    fn shortest_path(&self, n1: Node, n2: Node) -> Cost {
        match self.get(n1, n2) {
            PathCost::Path(c) => *c,
            _ => panic!("There is no shortest path known between {} and {}.", n1, n2),
        }
    }
}

#[cfg(test)]
mod test_state {
    use crate::graph::AdjListGraph;

    use super::*;

    fn get_graph() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 2.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());

        graph
    }

    ///   1 --5-- 2 --1-- 3
    ///  |2|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    #[test]
    fn test_state_init_all_pairs() {
        let graph = get_graph();
        let state = State::cached(&graph, 1.into());

        assert_eq!(
            state.visible_nodes().to_sorted_vec(),
            vec![1.into(), 2.into(), 4.into()]
        );
        assert_eq!(state.explored.clone().to_sorted_vec(), vec![1.into()]);
        assert_eq!(
            state.explored_boundary.clone().to_sorted_vec(),
            vec![1.into()]
        );
        assert_eq!(
            state.unexplored_boundary.clone().to_sorted_vec(),
            vec![2.into(), 4.into()]
        );

        assert_eq!(Cost::from(5), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(2), state.sp.shortest_path(1.into(), 4.into()));
        assert_eq!(Cost::from(7), state.sp.shortest_path(4.into(), 2.into()));
    }

    #[test]
    fn test_state_init_lazy() {
        let graph = get_graph();
        let state = State::lazy(&graph, 1.into());

        assert_eq!(
            state.visible_nodes().to_sorted_vec(),
            vec![1.into(), 2.into(), 4.into()]
        );
        assert_eq!(state.explored.clone().to_sorted_vec(), vec![1.into()]);
        assert_eq!(
            state.explored_boundary.clone().to_sorted_vec(),
            vec![1.into()]
        );
        assert_eq!(
            state.unexplored_boundary.clone().to_sorted_vec(),
            vec![2.into(), 4.into()]
        );

        assert_eq!(Cost::from(5), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(2), state.sp.shortest_path(1.into(), 4.into()));
        assert_eq!(Cost::from(7), state.sp.shortest_path(4.into(), 2.into()));
    }

    ///   1 --5-- 2 --1-- 3
    ///  |2|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    #[test]
    fn test_update_boundary_pairs() {
        let graph = get_graph();
        let mut state = State::cached(&graph, 1.into());

        state.move_sp_to(4.into());

        assert_eq!(
            state.visible_nodes().to_sorted_vec(),
            vec![1.into(), 2.into(), 4.into(), 5.into()]
        );
        assert_eq!(
            state.explored.clone().to_sorted_vec(),
            vec![1.into(), 4.into()]
        );
        assert_eq!(
            state.explored_boundary.clone().to_sorted_vec(),
            vec![1.into(), 4.into()]
        );
        assert_eq!(
            state.unexplored_boundary.clone().to_sorted_vec(),
            vec![2.into(), 5.into()]
        );

        assert_eq!(Cost::from(5), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(2), state.sp.shortest_path(1.into(), 4.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(1.into(), 5.into()));
        assert_eq!(Cost::from(8), state.sp.shortest_path(2.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(4.into(), 5.into()));

        state.move_sp_to(5.into());
        // 4 is not relevant anymore
        assert_eq!(
            state.visible_nodes().to_sorted_vec(),
            vec![1.into(), 2.into(), 4.into(), 5.into(), 6.into()]
        );
        assert_eq!(
            state.explored.clone().to_sorted_vec(),
            vec![1.into(), 4.into(), 5.into()]
        );
        assert_eq!(
            state.explored_boundary.clone().to_sorted_vec(),
            vec![1.into(), 5.into()]
        );
        assert_eq!(
            state.unexplored_boundary.clone().to_sorted_vec(),
            vec![2.into(), 6.into()]
        );

        assert_eq!(Cost::from(4), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(1.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(2.into(), 5.into()));
        assert_eq!(Cost::from(7), state.sp.shortest_path(2.into(), 6.into()));

        state.move_sp_to(2.into());
        // 1 is not relevant anymore
        assert_eq!(
            state.visible_nodes().to_sorted_vec(),
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );
        assert_eq!(
            state.explored.clone().to_sorted_vec(),
            vec![1.into(), 2.into(), 4.into(), 5.into()]
        );
        assert_eq!(
            state.explored_boundary.clone().to_sorted_vec(),
            vec![2.into(), 5.into()]
        );
        assert_eq!(
            state.unexplored_boundary.clone().to_sorted_vec(),
            vec![3.into(), 6.into()]
        );

        assert_eq!(Cost::from(1), state.sp.shortest_path(2.into(), 5.into()));
        assert_eq!(Cost::from(6), state.sp.shortest_path(6.into(), 5.into()));
        assert_eq!(Cost::from(8), state.sp.shortest_path(6.into(), 3.into()));

        state.move_sp_to(3.into());
        // 2 is not relevant anymore
        assert_eq!(
            state.visible_nodes().to_sorted_vec(),
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );
        assert_eq!(
            state.explored.clone().to_sorted_vec(),
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into()]
        );
        assert_eq!(
            state.explored_boundary.clone().to_sorted_vec(),
            vec![3.into(), 5.into()]
        );
        assert_eq!(
            state.unexplored_boundary.clone().to_sorted_vec(),
            vec![6.into()]
        );

        assert_eq!(6, state.visible_nodes().into_iter().count());
        assert_eq!(Cost::from(5), state.sp.shortest_path(6.into(), 5.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(6.into(), 3.into()));

        assert!(!state.graph_explored());

        state.move_sp_to(6.into());

        assert_eq!(
            state.visible_nodes().to_sorted_vec(),
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );
        assert_eq!(
            state.explored.clone().to_sorted_vec(),
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()]
        );
        assert_eq!(state.explored_boundary.clone().to_sorted_vec(), vec![]);
        assert_eq!(state.unexplored_boundary.clone().to_sorted_vec(), vec![]);

        assert!(state.graph_explored());
    }

    ///   1 --5-- 2 --1-- 3
    ///  |2|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    #[test]
    fn test_update_lazy() {
        let graph = get_graph();
        let mut state = State::lazy(&graph, 1.into());

        state.move_sp_to(4.into());
        assert_eq!(4, state.visible_nodes().into_iter().count());
        assert!(state.visible_nodes().contains(&1.into()));
        assert!(state.visible_nodes().contains(&2.into()));
        assert!(state.visible_nodes().contains(&4.into()));
        assert!(state.visible_nodes().contains(&5.into()));
        assert_eq!(Cost::from(5), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(2), state.sp.shortest_path(1.into(), 4.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(1.into(), 5.into()));
        assert_eq!(Cost::from(8), state.sp.shortest_path(2.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(4.into(), 5.into()));

        state.move_sp_to(5.into());
        assert_eq!(5, state.visible_nodes().into_iter().count());
        assert_eq!(Cost::from(4), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(2), state.sp.shortest_path(1.into(), 4.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(1.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(2.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(4.into(), 5.into()));
        assert_eq!(Cost::from(7), state.sp.shortest_path(2.into(), 6.into()));

        state.move_sp_to(2.into());
        assert_eq!(6, state.visible_nodes().into_iter().count());
        assert_eq!(Cost::from(4), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(2), state.sp.shortest_path(1.into(), 4.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(1.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(2.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(4.into(), 5.into()));
        assert_eq!(Cost::from(6), state.sp.shortest_path(6.into(), 5.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(4.into(), 3.into()));
        assert_eq!(Cost::from(8), state.sp.shortest_path(6.into(), 3.into()));

        state.move_sp_to(3.into());
        assert_eq!(6, state.visible_nodes().into_iter().count());
        assert_eq!(Cost::from(4), state.sp.shortest_path(1.into(), 2.into()));
        assert_eq!(Cost::from(2), state.sp.shortest_path(1.into(), 4.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(1.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(2.into(), 5.into()));
        assert_eq!(Cost::from(1), state.sp.shortest_path(4.into(), 5.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(4.into(), 3.into()));
        assert_eq!(Cost::from(5), state.sp.shortest_path(6.into(), 5.into()));
        assert_eq!(Cost::from(3), state.sp.shortest_path(6.into(), 3.into()));

        assert!(!state.graph_explored());

        state.move_sp_to(6.into());

        assert!(state.graph_explored());
    }
}
