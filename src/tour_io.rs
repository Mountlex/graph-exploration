use std::{
    error::Error,
    fs::File,
    io::BufReader,
    io::{BufRead, BufWriter, Write},
    path::PathBuf,
};

use graph_explore::graph::{AdjListGraph, GraphSize, Node, NodeSet, ShortestPaths, Tour};

pub fn import_tour<G>(filename: &PathBuf, sp: &G) -> Result<Tour, Box<dyn Error>>
where
    G: ShortestPaths,
{
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let ids: Vec<Node> = reader
        .lines()
        .map(|line| line.unwrap().parse::<usize>().unwrap())
        .map(|n| Node::new(n))
        .collect();
    Ok(Tour::with_cost_from(ids, sp))
}

pub fn check_tour(graph: &AdjListGraph, tour: &Tour) -> bool {
    let ns = tour.into_iter().copied().collect::<NodeSet>();
    return (graph.n() == ns.len()) && (graph.n() == tour.len() - 1);
}

pub fn export_tour(filename: &PathBuf, tour: &Tour) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);

    for node in tour {
        writeln!(writer, "{}", node.id())?;
    }
    writer.flush()?;

    Ok(())
}
