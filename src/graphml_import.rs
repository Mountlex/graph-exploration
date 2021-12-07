use std::path::PathBuf;

use graph_explore::cost::Cost;
use graph_explore::graph;

use std::collections::BTreeMap;

pub fn graphml_import(filename: PathBuf, max_nodes: Option<usize>) -> graph::AdjListGraph {
    let mut graph = graph::AdjListGraph::new();

    let text = std::fs::read_to_string(filename).unwrap();
    match roxmltree::Document::parse(&text) {
        Ok(doc) => {
            let length_key = doc
                .descendants()
                .find(|n| {
                    n.tag_name().name() == "key" && n.attribute("attr.name") == Some("length")
                })
                .unwrap()
                .attribute("id")
                .unwrap();

            if let Some(g) = doc.descendants().find(|n| n.tag_name().name() == "graph") {
                let mut ids = BTreeMap::<u64, usize>::new();

                for (i, vertex) in g
                    .descendants()
                    .filter(|n| n.tag_name().name() == "node")
                    .enumerate()
                {
                    let id = i;
                    let osm_id = vertex.attribute("id").unwrap().parse::<u64>().unwrap();
                    ids.insert(osm_id, id);
                }

                for edge in g.descendants().filter(|n| n.tag_name().name() == "edge") {
                    let source_osm = edge.attribute("source").unwrap().parse::<u64>().unwrap();
                    let sink_osm = edge.attribute("target").unwrap().parse::<u64>().unwrap();

                    let source = *ids.get(&source_osm).unwrap();
                    let sink = *ids.get(&sink_osm).unwrap();

                    if source < sink && (max_nodes.is_none() || sink < max_nodes.unwrap()) {
                        if let Some(length_node) = edge.descendants().find(|c| {
                            c.tag_name().name() == "data" && c.attribute("key") == Some(length_key)
                        }) {
                            let real = length_node.text().unwrap().trim().parse::<f64>().unwrap();
                            let cost = Cost::new(real.ceil() as usize);
                            graph.add_edge(source.into(), sink.into(), cost);
                        }
                    }
                }
            }
        }
        Err(e) => println!("Error: {}.", e),
    }

    graph
}
