use std::path::PathBuf;

use graph_explore::cost::Cost;
use graph_explore::graph;

pub fn xml_import(filename: PathBuf, max_nodes: Option<usize>) -> graph::AdjListGraph {
    let mut graph = graph::AdjListGraph::new();

    let text = std::fs::read_to_string(filename).unwrap();
    match roxmltree::Document::parse(&text) {
        Ok(doc) => {
            if let Some(g) = doc.descendants().find(|n| n.tag_name().name() == "graph") {
                for (i, vertex) in g
                    .descendants()
                    .filter(|n| n.tag_name().name() == "vertex")
                    .enumerate()
                {
                    let id = i;
                    for edge in vertex
                        .descendants()
                        .filter(|n| n.tag_name().name() == "edge")
                    {
                        let sink_id = edge
                            .first_child()
                            .unwrap()
                            .text()
                            .unwrap()
                            .parse::<usize>()
                            .unwrap();

                        if id < sink_id && (max_nodes.is_none() || sink_id < max_nodes.unwrap()) {
                            let real = edge.attribute("cost").unwrap().parse::<f64>().unwrap();
                            let cost = Cost::new(real.ceil() as usize);
                            graph.add_edge(id.into(), sink_id.into(), cost);
                        }
                    }
                }
            }
        }
        Err(e) => println!("Error: {}.", e),
    }

    graph
}
