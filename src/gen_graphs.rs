use graph_explore::{
    cost::Cost,
    graph::{AdjListGraph, Edge, Edges, GraphSize, Node, Nodes, Tour},
};
use rand::Rng;

pub struct CostConfig {
    min: Cost,
    max: Cost,
}

pub struct RandomGraphGenerator {
    n: usize,
    density: f64,
    cost_config: CostConfig,
}

impl RandomGraphGenerator {
    pub fn default_costs(n: usize, density: f64) -> Self {
        RandomGraphGenerator {
            n,
            density,
            cost_config: CostConfig {
                min: 1.into(),
                max: 2000.into(),
            },
        }
    }

    fn random_cost(&self) -> Cost {
        let mut rng = rand::thread_rng();
        rng.gen_range(self.cost_config.min..self.cost_config.max)
    }
}

impl Iterator for RandomGraphGenerator {
    type Item = AdjListGraph;

    fn next(&mut self) -> Option<Self::Item> {
        let mut num_edges: u64 = 0;

        let mut rng = rand::thread_rng();

        let mut graph = AdjListGraph::new();
        for (n1, n2) in (0..self.n - 1).zip(1..self.n) {
            graph.add_edge(n1.into(), n2.into(), self.random_cost());
            num_edges += 1;
        }

        while ((2 * num_edges) as f64) < (self.density * (self.n * self.n) as f64) {
            let source = rng.gen_range(0..self.n);
            let sink = rng.gen_range(0..self.n);
            graph.add_edge(source.into(), sink.into(), self.random_cost());
            num_edges += 1;
        }

        Some(graph)
    }
}

pub fn generate_rosenkrantz(i: usize) -> (AdjListGraph, Tour) {
    let scaling_param = 1;
    let mut graph: AdjListGraph = generate_rosenkrantz_rec(i, scaling_param);
    let mut nodes: Vec<Node> = graph.nodes().collect();
    nodes.sort();
    graph.add_edge(
        *nodes.first().unwrap(),
        *nodes.last().unwrap(),
        (scaling_param + 3).into(),
    );
    nodes.push(*nodes.first().unwrap());
    let tour = Tour::with_cost_from(nodes, &graph);
    (graph, tour)
}

fn generate_rosenkrantz_rec(i: usize, scaling_param: usize) -> AdjListGraph {
    let mut graph = AdjListGraph::new();
    if i == 1 {
        graph.add_edge(1.into(), 3.into(), scaling_param.into());
        graph.add_edge(3.into(), 2.into(), scaling_param.into());
        graph.add_edge(1.into(), 2.into(), (scaling_param + 1).into());
    } else {
        graph = generate_rosenkrantz_rec(i - 1, scaling_param);
        let size = graph.n();
        let edges: Vec<Edge> = graph.edges().collect();
        for e in edges {
            graph.add_edge(
                (e.source().id() + size + 1).into(),
                (e.sink().id() + size + 1).into(),
                e.cost(),
            );
        }
        let mid_1 = size / 2 + 1;
        let mid_2 = mid_1 + size + 1;
        let cost_chord =
            scaling_param * (4 * usize::pow(2, (i - 1) as u32) - (i % 2) + ((i + 1) % 2) + 3) / 6;
        graph.add_edge(size.into(), (size + 1).into(), (scaling_param + 1).into());
        graph.add_edge(
            (size + 1).into(),
            (size + 2).into(),
            (scaling_param + 1).into(),
        );
        graph.add_edge(mid_1.into(), (size + 2).into(), cost_chord.into());
        graph.add_edge(mid_2.into(), (size + 1).into(), cost_chord.into());
    }
    graph
}

pub fn generate_blocking_lb(i: usize, delta: f32) -> (AdjListGraph, Tour) {
    let scaling_param = 100;
    let one_plus_delta_rounded_scaled: usize =
        (scaling_param as f32 * delta) as usize + scaling_param + 1;
    let n = usize::pow(2, i as u32);
    let mut graph = AdjListGraph::new();
    let mut current: usize = 2;
    let mut bot = Vec::new();
    for i in 0..(n - 1) {
        bot.push(4 * i + 3);
    }
    for _ in 0..4 * (n - 1) {
        if (current % 4) == 0 {
            graph.add_edge(
                (current - 3).into(),
                current.into(),
                one_plus_delta_rounded_scaled.into(),
            );
        } else {
            graph.add_edge(
                (current - 1).into(),
                current.into(),
                one_plus_delta_rounded_scaled.into(),
            );
        }
        current += 1;
    }
    for _ in 0..n {
        graph.add_edge((current - 1).into(), current.into(), scaling_param.into());
        current += 1;
    }
    for k in 1..i + 1 {
        for l in 0..usize::pow(2, (i - k) as u32) {
            graph.add_edge((current - 1).into(), current.into(), scaling_param.into());
            graph.add_edge(
                (current - 1).into(),
                bot.remove(l).into(),
                (scaling_param * usize::pow(2, k as u32)).into(),
            );
            current += 1;
        }
        for _ in
            0..(one_plus_delta_rounded_scaled) * (usize::pow(2, (k + 2) as u32)) / scaling_param + 1
        {
            if k == i {
                break;
            }
            graph.add_edge((current - 1).into(), current.into(), scaling_param.into());
            current += 1;
        }
    }

    let mut nodes: Vec<Node> = graph.nodes().collect();
    nodes.sort();
    graph.add_edge(
        *nodes.first().unwrap(),
        *nodes.last().unwrap(),
        (n * one_plus_delta_rounded_scaled + 1).into(),
    );
    nodes.push(*nodes.first().unwrap());
    let tour = Tour::with_cost_from(nodes, &graph);

    (graph, tour)
}

#[cfg(test)]
mod test_generation {
    use super::*;

    #[test]
    fn test_blocking_lb() {
        let (graph, _) = generate_blocking_lb(3, 1.2);
        for e in graph.edges() {
            println!("{} {} {}", e.source(), e.sink(), e.cost());
        }
    }
}

/* #[test]
fn test_rosenkranz() {
    let (graph, _) = generate_rosenkrantz(4);
    for e in graph.edges() {
        println!("{} {} {}", e.source(), e.sink(), e.cost());
    }
} */
