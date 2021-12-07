use std::{collections::HashMap, fs::File, path::PathBuf};

use geo::point;
use geo::prelude::*;
use graph_explore::cost::Cost;
use graph_explore::graph::{self, AdjListGraph};
use osmpbfreader::NodeId;

pub fn pbf_import(path: PathBuf) -> AdjListGraph {
    let file = File::open(path).unwrap();
    let mut pbf = osmpbfreader::OsmPbfReader::new(file);
    let mut graph = AdjListGraph::new();

    let mut nodes = HashMap::<NodeId, (f64, f64)>::new();
    for obj in pbf.iter() {
        let obj = obj.unwrap();

        if let Some(node) = obj.node() {
            if node.id.0 < 0 {
                println!("nodeid < 0!");
            }
            nodes.insert(node.id, (node.lat(), node.lon()));
        }
    }

    let mut way_count: HashMap<NodeId, usize> = HashMap::new();
    pbf.rewind().unwrap();
    for obj in pbf.iter() {
        let obj = obj.unwrap();
        if let Some(way) = obj.way() {
            if way.tags.contains_key("highway") && way.nodes.len() > 1 {
                for &node in &way.nodes {
                    let count = way_count.entry(node).or_insert(0);
                    *count += 1;
                }
            }
        }
    }

    let valid_highways = vec!["motorway"]; //, "trunk"];//, "primary", "secondary", "tertiary", "living_street"];//, "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"];
    pbf.rewind().unwrap();
    for obj in pbf.iter() {
        let obj = obj.unwrap();
        if let Some(way) = obj.way() {
            if way.tags.contains_key("highway")
                && valid_highways.contains(&way.tags.get("highway").unwrap().as_str())
                && way.nodes.len() > 1
            {
                let mut start = way.nodes[0];
                let mut total_dist: f64 = 0.0;
                for window in way.nodes.windows(2) {
                    let source_id = window[0];
                    let sink_id = window[1];
                    let (source_lat, source_lon) = nodes.get(&source_id).unwrap();
                    let (sink_lat, sink_lon) = nodes.get(&sink_id).unwrap();
                    let source_p = point!(x: *source_lon, y: *source_lat);
                    let sink_p = point!(x: *sink_lon, y: *sink_lat);
                    let dist = source_p.geodesic_distance(&sink_p);
                    total_dist += dist;

                    if *way_count.get(&sink_id).unwrap() > 1 {
                        // intersection at sink
                        let source = graph::Node::new(start.0 as usize);
                        let sink = graph::Node::new(sink_id.0 as usize);
                        let cost = Cost::new(total_dist.round() as usize);
                        graph.add_edge(source, sink, cost);

                        total_dist = 0.0;
                        start = sink_id;
                    }
                }
            }
        }
    }

    graph
}
