use ndarray::Array2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    dijkstra::shortest_paths_to,
    graph::{Graph, Node, NodeIndex, Nodes, ShortestPaths},
    Cost,
};

impl ShortestPaths for ShortestPathsCache {
    fn shortest_path_cost(&self, n1: Node, n2: Node) -> Cost {
        self.get(n1, n2)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum PathCost {
    Unreachable,
    Path(Cost),
}

#[derive(Debug, Clone)]
pub struct ShortestPathsCache {
    matrix: Array2<PathCost>,
    index: NodeIndex,
}

impl ShortestPathsCache {
    pub fn empty<'a, G>(graph: &'a G) -> Self
    where
        G: Nodes<'a>,
    {
        let mut nodes = graph.nodes().collect::<Vec<Node>>();
        nodes.sort();
        let n = nodes.len();
        let mut d = Array2::from_elem((n, n), PathCost::Unreachable);
        for i in 0..n {
            d[[i, i]] = PathCost::Path(0.into());
        }

        ShortestPathsCache {
            matrix: d,
            index: NodeIndex::init(&nodes),
        }
    }

    pub fn compute_all_pairs<'a, G>(graph: &'a G) -> Self
    where
        G: Graph<'a>,
    {
        log::info!("Starting to compute all pair shortest paths.");

        let mut nodes = graph.nodes().collect::<Vec<Node>>();
        nodes.sort();
        let n = nodes.len();
        let mut d = Array2::from_elem((n, n), PathCost::Unreachable);
        for i in 0..n {
            d[[i, i]] = PathCost::Path(0.into());
        }

        let mut sp = ShortestPathsCache {
            matrix: d,
            index: NodeIndex::init(&nodes),
        };

        for (i, &n1) in nodes.iter().enumerate() {
            log::trace!("Start node {}/{}", i, n);

            let goals: &[Node] = nodes.split_at(i).1;
            let paths = shortest_paths_to(graph, n1, &goals);
            for &n in goals {
                sp.set(n1, n, paths.cost_to(n).unwrap());
            }
        }

        log::info!("Finished computing all pair shortest paths!");

        sp
    }

    pub fn compute_all_pairs_par<'a, G>(graph: &'a G) -> Self
    where
        G: Graph<'a> + Sync,
    {
        log::info!("Starting to compute all pair shortest paths (parallel).");

        let mut nodes = graph.nodes().collect::<Vec<Node>>();
        nodes.sort();
        let n = nodes.len();
        let mut d = Array2::from_elem((n, n), PathCost::Unreachable);
        for i in 0..n {
            d[[i, i]] = PathCost::Path(0.into());
        }

        let mut sp = ShortestPathsCache {
            matrix: d,
            index: NodeIndex::init(&nodes),
        };

        let indexed_nodes: Vec<(usize, Node)> = nodes.iter().copied().enumerate().collect();
        let costs = indexed_nodes
            .into_par_iter()
            .map(|(i, n)| {
                log::trace!("Start node {}/{}", i, n);
                let goals: &[Node] = nodes.split_at(i).1;
                let paths = shortest_paths_to(graph, n, &goals);
                (
                    i,
                    n,
                    goals
                        .into_iter()
                        .map(|g| paths.cost_to(*g).unwrap())
                        .collect::<Vec<Cost>>(),
                )
            })
            .collect::<Vec<(usize, Node, Vec<Cost>)>>();

        for (i, n1, c) in costs {
            let goals: &[Node] = nodes.split_at(i).1;
            for (n, cost) in goals.into_iter().zip(c) {
                sp.set(n1, *n, cost);
            }
        }
        log::info!("Finished computing all pair shortest paths!");

        sp
    }

    pub fn get_or_compute<'a, G>(&mut self, n1: Node, n2: Node, graph: &'a G) -> Cost
    where
        G: Graph<'a>,
    {
        let i1 = self.index[&n1];
        let i2 = self.index[&n2];

        let x = i1.min(i2);
        let y = i1.max(i2);
        if let PathCost::Path(cost) = self.matrix[[x, y]] {
            cost
        } else if let Some(cost) = graph.edge_cost(n1, n2) {
            self.matrix[[x, y]] = PathCost::Path(cost);
            cost
        } else {
            let goals: Vec<Node> = graph.nodes().collect();
            let paths = shortest_paths_to(graph, n1, &goals);
            for n in goals {
                self.set(n1, n, paths.cost_to(n).unwrap());
            }
            if let PathCost::Path(cost) = self.matrix[[x, y]] {
                cost
            } else {
                panic!("Should not happen")
            }
        }
    }

    pub fn get(&self, n1: Node, n2: Node) -> Cost {
        let i1 = self.index[&n1];
        let i2 = self.index[&n2];

        let x = i1.min(i2);
        let y = i1.max(i2);
        if let PathCost::Path(cost) = self.matrix[[x, y]] {
            cost
        } else {
            panic!("No path known!")
        }
    }

    fn set(&mut self, n1: Node, n2: Node, cost: Cost) {
        let i1 = self.index[&n1];
        let i2 = self.index[&n2];

        let x = i1.min(i2);
        let y = i1.max(i2);

        self.matrix[[x, y]] = PathCost::Path(cost);
    }
}
