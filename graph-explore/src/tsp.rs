use std::process::Command;
use std::{
    collections::{HashMap, HashSet},
    num::ParseIntError,
    str::FromStr,
};
use std::{fs, io::BufWriter};

use lp_modeler::dsl::*;
use lp_modeler::solvers::{GurobiSolver, SolverTrait, Status};
use rayon::prelude::*;

use rand::seq::SliceRandom;
use std::io::{BufRead, Write};

use crate::{
    cost::Cost,
    graph::{
        AdjListGraph, Adjacency, Edge, Graph, GraphSize, MultiGraph, Node, NodeSet, TotalWeight,
        Tour, Tree,
    },
    sp::ShortestPathsCache,
    two_opt::TwoOpt,
};
use crate::{dijkstra::shortest_paths_to, graph::DetStartNode};

fn compute_odd_vertices<'a, G>(graph: &'a G) -> Vec<Node>
where
    G: Graph<'a>,
{
    graph
        .nodes()
        .into_iter()
        .filter(|&n| graph.adjacent(n).count() % 2 == 1)
        .collect::<Vec<Node>>()
}

fn artifical_edges_to(graph: &AdjListGraph, node: Node, sinks: &[Node]) -> Vec<Edge> {
    let mut edges = vec![];
    let paths = shortest_paths_to(graph, node, sinks);
    for &n2 in sinks {
        if n2 != node {
            let cost = paths.cost_to(n2).unwrap();
            let edge = Edge::new(node, n2, cost);
            assert!(edge.sink() != edge.source());
            edges.push(edge);
        }
    }
    edges
}

fn blossom_algorithm(graph: &AdjListGraph, odd_vertices: &[Node]) -> Vec<Edge> {
    log::info!("Using Edmond's blossom algorithm to compute optimal matching.");

    let input_file = &format!(
        "input_{}.txt",
        chrono::Local::now().format("%d%m%Y-%H%M-%f")
    );
    let output_file = &format!(
        "output_{}.txt",
        chrono::Local::now().format("%d%m%Y-%H%M-%f")
    );

    let mut nodes = odd_vertices.to_owned();
    nodes.sort();
    let min_node_index = nodes.first().unwrap().id();
    let max_node_index = nodes.last().unwrap().id();
    let mut index: Vec<usize> = vec![0; max_node_index - min_node_index + 1];
    for (i, node) in nodes.iter().enumerate() {
        index[node.id() - min_node_index] = i;
    }

    log::info!("Computing shortest path completion between odd vertices.");
    let edges: Vec<Edge> = odd_vertices
        .par_iter()
        .flat_map(|n1| {
            let larger = odd_vertices
                .into_iter()
                .filter(|n| n1 < n)
                .copied()
                .collect::<Vec<Node>>();
            artifical_edges_to(graph, *n1, &larger)
        })
        .collect();

    log::info!("Writing matching data to file.");

    let input = fs::File::create(input_file).expect("Could not create file");
    let mut writer = BufWriter::new(input);
    writeln!(writer, "{} {}", nodes.len(), edges.len()).unwrap();
    for e in &edges {
        let n1 = index[e.source().id() - min_node_index];
        let n2 = index[e.sink().id() - min_node_index];
        writeln!(writer, "{} {} {}", n1, n2, e.cost().as_float()).unwrap();
    }
    writer.flush().unwrap();

    log::info!("Calling Blossom V algorithm...");
    if let Ok(res) = Command::new("./blossom/blossom5")
        .arg("-e")
        .arg(input_file)
        .arg("-w")
        .arg(output_file)
        .output()
    {
        log::info!(
            "Finished! Blossom V output: {}",
            String::from_utf8(res.stdout).unwrap()
        );
        let output = fs::File::open(output_file).unwrap();
        let reader = std::io::BufReader::new(output);

        let mut matching = vec![0; nodes.len()];
        for line in reader.lines().skip(1) {
            let edge_nodes: Vec<usize> = line
                .unwrap()
                .split(" ")
                .map(|s| s.parse::<usize>().unwrap())
                .collect();
            let n1 = edge_nodes[0];
            let n2 = edge_nodes[1];
            matching[n1] = n2;
            matching[n2] = n1;
        }

        log::info!("Removing data files.");

        fs::remove_file(input_file).unwrap();
        fs::remove_file(output_file).unwrap();

        edges
            .into_iter()
            .filter(|e| {
                let n1 = index[e.source().id() - min_node_index];
                let n2 = index[e.sink().id() - min_node_index];
                matching[n1] == n2
            })
            .collect()
    } else {
        fs::remove_file(input_file).unwrap();
        vec![]
    }
}

fn greedy_matching(graph: &AdjListGraph, odd_vertices: &[Node]) -> Vec<Edge> {
    log::info!("Using a greedy algorithm to compute approximation.");

    let mut nodes = odd_vertices.to_owned();
    let mut rng = rand::thread_rng();

    let edges = odd_vertices
        .iter()
        .map(|n| (*n, artifical_edges_to(graph, *n, &odd_vertices)))
        .collect::<HashMap<Node, Vec<Edge>>>();

    let mut current_best: Option<(Cost, Vec<Edge>)> = None;
    for _ in 0..10 {
        let mut matched = HashSet::<Node>::new();
        let mut cost: Cost = 0.into();
        let mut res: Vec<Edge> = vec![];
        nodes.shuffle(&mut rng);
        for n1 in &nodes {
            if !matched.contains(&n1) {
                matched.insert(*n1);
                let art_edges: Vec<Edge> = edges
                    .get(n1)
                    .unwrap()
                    .into_iter()
                    .filter(|e| !matched.contains(&e.sink()))
                    .cloned()
                    .collect();
                let e = art_edges.iter().min_by_key(|e| e.cost()).unwrap();
                matched.insert(e.sink());
                cost += e.cost();
                res.push(e.clone());
            }
        }
        if let Some((c, best)) = &mut current_best {
            if cost < *c {
                *best = res;
                *c = cost;
            }
        } else {
            current_best = Some((cost, res));
        }
    }

    current_best.unwrap().1
}

fn min_cost_perfect_matching(graph: &AdjListGraph, odd_vertices: &[Node]) -> Vec<Edge> {
    log::info!("Using an ILP solver to compute optimal matching.");

    let mut model = LpProblem::new("matching", LpObjective::Minimize);
    let mut edges: Vec<Edge> = vec![];

    for n1 in odd_vertices {
        let larger = odd_vertices
            .into_iter()
            .filter(|n| n1 < n)
            .copied()
            .collect::<Vec<Node>>();
        edges.append(&mut artifical_edges_to(graph, *n1, &larger));
    }

    log::info!("Min-Cost perfect matching: {} edges", edges.len());

    if edges.is_empty() {
        log::warn!("There are no edges. Break min cost matching.");
        return edges;
    }

    let vars: Vec<(&Edge, LpBinary)> = edges
        .iter()
        .enumerate()
        .map(|(i, edge)| (edge, LpBinary::new(&format!("x{}", i))))
        .collect();

    let obj_vec: Vec<LpExpression> = vars
        .iter()
        .map(|(e, var)| (e.cost().as_float() as f32) * var)
        .collect();
    model += obj_vec.sum();

    for &node in odd_vertices {
        let constr: Vec<LpExpression> = vars
            .iter()
            .filter(|(e, _)| e.sink() == node || e.source() == node)
            .map(|(_, var)| 1.0 * var)
            .collect();
        model += constr.sum().equal(1.0);
    }

    let solver = GurobiSolver::new();
    //let solver = lp_modeler::solvers::CbcSolver::new();
    match solver.run(&model) {
        Ok(solution) => match solution.status {
            Status::Optimal => vars
                .into_iter()
                .filter(|(_, var)| match solution.results.get(&var.name) {
                    Some(&value) => (value - 1.0).abs() < f32::EPSILON,
                    None => false, // Isolated verticies are not in the result map
                })
                .map(|(edge, _)| edge.clone())
                .collect::<Vec<Edge>>(),
            _ => panic!("Min-cost matching: No optimal solution"),
        },
        Err(msg) => panic!("Min-cost matching solver error: {}", msg),
    }
}

fn eulerian_path(start: Node, graph: &MultiGraph) -> Vec<Node> {
    if !graph.contains_node(start) {
        return vec![];
    }

    let mut start_node_index: usize = 0;
    let mut start_node = start;
    let mut circle: Vec<Node> = vec![start_node];

    let mut current = start_node;
    let mut current_circle_len: usize = 0;

    let mut w_graph = graph.clone();

    loop {
        // build up cycle
        let mut adj = w_graph.adjacent(current).collect::<Vec<Edge>>();
        adj.sort_by_key(|e| e.sink());
        if let Some(edge) = adj.first() {
            current = edge.sink();
            w_graph.remove_edge_raw(*edge);
            current_circle_len += 1;
            circle.insert(start_node_index + current_circle_len, current);
        } else {
            assert!(current == start_node);
            break;
        }

        if current == start_node {
            current_circle_len = 0;
            for (idx, node) in circle.iter().enumerate() {
                if w_graph.adjacent(*node).count() > 0 {
                    start_node_index = idx;
                    start_node = *node;
                    current = start_node;
                    break;
                }
            }
        }
    }

    circle
}

fn construct_tour(eulerian_tour: Vec<Node>, multi_graph: &MultiGraph) -> Tour {
    assert_eq!(eulerian_tour.first(), eulerian_tour.last());
    let mut tour: Vec<Node> = vec![];
    let mut cost: Cost = 0.into();
    let mut prev: Option<Node> = None;
    let mut visited = NodeSet::empty();

    for node in eulerian_tour {
        if let Some(prev) = &mut prev {
            if !visited.contains(&node) {
                tour.push(node);
                visited.insert(node);
            }
            cost += multi_graph.edge_cost(node, *prev).unwrap();
            *prev = node;
        } else {
            tour.push(node);
            visited.insert(node);
            prev = Some(node);
        }
    }
    tour.push(*tour.first().unwrap());

    Tour::new(tour, cost)
}

#[derive(Debug, Copy, Clone)]
pub enum MatchingAlgorithm {
    ILP,
    Greedy,
    Blossom,
}

impl Default for MatchingAlgorithm {
    fn default() -> Self {
        MatchingAlgorithm::Blossom
    }
}

impl FromStr for MatchingAlgorithm {
    type Err = ParseIntError;
    fn from_str(day: &str) -> Result<Self, Self::Err> {
        match day {
            "ilp" => Ok(MatchingAlgorithm::ILP),
            "greedy" => Ok(MatchingAlgorithm::Greedy),
            "blossom" => Ok(MatchingAlgorithm::Blossom),
            _ => panic!("Could not parse matching algorithm"),
        }
    }
}

pub fn christofides(
    graph: &AdjListGraph,
    mst: &Tree,
    sp: &ShortestPathsCache,
    matching_algo: MatchingAlgorithm,
    max_iterations: Option<usize>,
) -> Tour {
    log::info!("Start computing TSP approximation by Christofides algorithm.");

    let odd_vertices = compute_odd_vertices(mst);

    assert!(odd_vertices.len() % 2 == 0);

    log::info!(
        "Computing a min-cost perfect matching on {} vertices.",
        odd_vertices.len()
    );
    let min_cost_matching = match matching_algo {
        MatchingAlgorithm::ILP => min_cost_perfect_matching(graph, &odd_vertices),
        MatchingAlgorithm::Greedy => greedy_matching(graph, &odd_vertices),
        MatchingAlgorithm::Blossom => blossom_algorithm(graph, &odd_vertices),
    };
    assert!(odd_vertices.len() == 2 * min_cost_matching.len());

    let mut multi_graph = MultiGraph::from_graph(mst);
    let mut matching_cost: Cost = 0.into();
    for edge in &min_cost_matching {
        matching_cost += edge.cost();
        multi_graph.add_edge_raw(*edge);
    }
    assert!(compute_odd_vertices(&multi_graph).is_empty());

    log::info!("MST value: {}", mst.total_weight());
    log::info!("Min-Cost perfect matching value: {}", matching_cost);

    let eulerian_tour = eulerian_path(graph.start_node(), &multi_graph);
    let tour = construct_tour(eulerian_tour, &multi_graph);

    log::info!("Finished computing TSP approximation by Christofides algorithm.");

    if let Some(max_iterations) = max_iterations {
        log::info!("Start improving TSP approximation by 2-opt");
        let mut two_opt =
            TwoOpt::with_sp_cache(graph.n(), tour, sp, crate::two_opt::TwoOptType::Decrease);
        two_opt.run_for(max_iterations)
    } else {
        tour
    }
}

#[cfg(test)]
mod test_tsp {
    use super::*;
    use crate::graph::GraphSize;
    use crate::mst::prims_tree;

    ///   1 --5-- 2 --1-- 3
    ///  |3|     |1|     |3|
    ///   4 --1-- 5 --6-- 6
    fn get_graph1() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 5.into());
        graph.add_edge(1.into(), 4.into(), 3.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 6.into(), 3.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 6.into());
        graph
    }

    #[test]
    fn test_odd_mst1() {
        let graph = get_graph1();

        let (mst, _) = prims_tree(&graph);
        let mut odd_vertices = compute_odd_vertices(&mst);
        odd_vertices.sort();

        assert_eq!(odd_vertices, vec![1.into(), 6.into()]);
    }

    #[test]
    fn test_min_cost_matching1() {
        let graph = get_graph1();

        let (mst, _) = prims_tree(&graph);
        let odd_vertices = compute_odd_vertices(&mst);
        let matching = min_cost_perfect_matching(&graph, &odd_vertices);

        assert_eq!(matching.len(), 1);
        let edge = matching.first().unwrap();
        assert_eq!(edge.cost(), 9.into());
    }

    ///  https://de.wikipedia.org/wiki/Algorithmus_von_Christofides
    ///
    ///  1 ------- 2
    ///  |  \   /  |
    ///  |    3    |
    ///  |  /   \  |
    ///  4 ------- 5
    fn get_graph2() -> AdjListGraph {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(1.into(), 3.into(), 1.into());
        graph.add_edge(1.into(), 4.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
        graph.add_edge(3.into(), 5.into(), 1.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 4.into(), 2.into());
        graph.add_edge(1.into(), 5.into(), 2.into());
        graph
    }

    #[test]
    fn test_odd_mst2() {
        let mut mst = AdjListGraph::new();
        mst.add_edge(3.into(), 1.into(), 1.into());
        mst.add_edge(3.into(), 2.into(), 1.into());
        mst.add_edge(3.into(), 4.into(), 1.into());
        mst.add_edge(3.into(), 5.into(), 1.into());
        let mut odd_vertices = compute_odd_vertices(&mst);
        odd_vertices.sort();

        assert_eq!(odd_vertices, vec![1.into(), 2.into(), 4.into(), 5.into()]);
    }

    #[test]
    fn test_min_cost_matching2() {
        let graph = get_graph2();
        let mut mst = AdjListGraph::new();
        mst.add_edge(3.into(), 1.into(), 1.into());
        mst.add_edge(3.into(), 2.into(), 1.into());
        mst.add_edge(3.into(), 4.into(), 1.into());
        mst.add_edge(3.into(), 5.into(), 1.into());
        let odd_vertices = compute_odd_vertices(&mst);
        let matching = min_cost_perfect_matching(&graph, &odd_vertices);

        assert_eq!(matching.len(), 2);
    }

    #[test]
    fn test_eulerian_graph2() {
        let graph = get_graph2();
        let mut mst = AdjListGraph::new();
        mst.add_edge(3.into(), 1.into(), 1.into());
        mst.add_edge(3.into(), 2.into(), 1.into());
        mst.add_edge(3.into(), 4.into(), 1.into());
        mst.add_edge(3.into(), 5.into(), 1.into());
        let odd_vertices = compute_odd_vertices(&mst);
        let matching = min_cost_perfect_matching(&graph, &odd_vertices);

        let mut multi_graph = MultiGraph::from_graph(&mst);
        for edge in matching {
            multi_graph.add_edge_raw(edge);
        }
        let euler_tour = eulerian_path(1.into(), &multi_graph);

        assert_eq!(euler_tour.len(), 7);
    }

    #[test]
    fn test_eulerian1() {
        let mut graph = MultiGraph::new();
        graph.add_edge(0.into(), 1.into(), 1.into());
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 0.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(5.into(), 6.into(), 1.into());
        graph.add_edge(6.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 7.into(), 1.into());
        graph.add_edge(7.into(), 8.into(), 1.into());
        graph.add_edge(8.into(), 9.into(), 1.into());
        graph.add_edge(8.into(), 10.into(), 1.into());
        graph.add_edge(8.into(), 10.into(), 1.into());
        graph.add_edge(9.into(), 3.into(), 1.into());

        let euler_tour = eulerian_path(0.into(), &graph);

        assert_eq!(euler_tour.len(), graph.m() + 1);
        assert_eq!(
            euler_tour,
            vec![
                0.into(),
                1.into(),
                2.into(),
                3.into(),
                7.into(),
                8.into(),
                10.into(),
                8.into(),
                9.into(),
                3.into(),
                4.into(),
                5.into(),
                6.into(),
                3.into(),
                0.into()
            ]
        );
    }

    #[test]
    fn test_eulerian2() {
        let mut graph = MultiGraph::new();
        graph.add_edge(0.into(), 1.into(), 1.into());
        graph.add_edge(1.into(), 0.into(), 1.into());

        let euler_tour = eulerian_path(0.into(), &graph);

        assert_eq!(euler_tour.len(), graph.m() + 1);
        assert_eq!(euler_tour, vec![0.into(), 1.into(), 0.into()]);
    }

    #[test]
    fn test_eulerian3() {
        let graph = MultiGraph::new();

        let euler_tour = eulerian_path(0.into(), &graph);
        assert!(euler_tour.is_empty());
    }

    #[test]
    fn test_eulerian4() {
        let graph = MultiGraph::singleton(0.into());
        let euler_tour = eulerian_path(0.into(), &graph);
        assert_eq!(euler_tour, vec![0.into()]);
    }

    ///  https://de.wikipedia.org/wiki/Algorithmus_von_Christofides
    ///
    ///  1 ------- 2
    ///  |  \   /  |
    ///  |    3    |
    ///  |  /   \  |
    ///  4 ------- 5
    #[test]
    fn test_tsp1() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(1.into(), 3.into(), 1.into());
        graph.add_edge(1.into(), 4.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(2.into(), 5.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
        graph.add_edge(3.into(), 5.into(), 1.into());
        graph.add_edge(4.into(), 5.into(), 1.into());
        graph.add_edge(2.into(), 4.into(), 2.into());
        graph.add_edge(1.into(), 5.into(), 2.into());

        let (mst, _) = prims_tree(&graph);
        assert_eq!(mst.n(), 5);
        let sp = ShortestPathsCache::compute_all_pairs(&graph);
        let tour = christofides(&graph, &mst, &sp, MatchingAlgorithm::ILP, None);
        assert_eq!(tour.len(), 6);
        assert_eq!(tour.first(), tour.last());
        assert_eq!(tour.cost(), 6.into());
    }
}
