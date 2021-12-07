use std::{error::Error, path::PathBuf};

use graph_explore::{
    algs::{robust_ptour, salesperson, theoretic_robust_ptour},
    graph::Tour,
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    import_from_file,
    sample_builder::SampleBuilder,
    samples::{export, AlgResult, ParamSample, PredRow, PredSample},
    tour_io::import_tour,
};

pub fn execute_city(name: String, theoretic: bool) -> Result<(), Box<dyn Error>> {
    let graph_path = PathBuf::from(format!("resources/osm/{}.graphml", name));
    let tour_path = PathBuf::from(format!("tours/{}.txt", name.to_lowercase()));
    let pred_path = PathBuf::from(format!("preds/{}", name.to_lowercase()));

    execute_city_with(
        &name.to_lowercase(),
        graph_path,
        tour_path,
        pred_path,
        theoretic,
    )
}

fn execute_city_with(
    name: &str,
    graph_path: PathBuf,
    tour_path: PathBuf,
    pred_path: PathBuf,
    theoretic: bool,
) -> Result<(), Box<dyn Error>> {
    let graph = import_from_file(graph_path.clone(), None).expect(&format!(
        "Cannot import graph! Should be located at {}",
        graph_path.to_str().unwrap()
    ));
    println!("Imported graph: ✔️");

    let sample = SampleBuilder::new()
        .with_imported_tour_or_default(Some(tour_path))
        .with_base_algorithms()
        .build(graph);
    println!("Base sample: ✔️");

    let paths: Vec<std::fs::DirEntry> = std::fs::read_dir(pred_path)?
        .filter_map(|e| e.ok())
        .collect();

    let preds: Vec<Tour> = paths
        .par_iter()
        .flat_map(|path| import_tour(&path.path(), &sample.sp))
        .collect();
    println!("Imported predictions: ✔️");

    let pred_samples: Vec<PredSample> = preds
        .into_par_iter()
        .map(|pred| {
            let tour = pred.to_neighbor_tour(&sample.graph);
            let mut results = vec![];
            results.push(AlgResult::new(
                "pTour".into(),
                salesperson(&sample.graph, &tour),
            ));
            PredSample::with_results(&sample, tour, results)
        })
        .collect();

    println!("Algorithms on predictions: ✔️");

    let params = vec![0.25, 0.5, 0.75, 1.0, 1.25];
    let param_samples: Vec<ParamSample<'_, PredSample, PredRow, f64>> = pred_samples
        .par_iter()
        .map(|sample| {
            params
                .iter()
                .map(|s| {
                    let mut results = vec![AlgResult::new(
                        "Robust_pTour".into(),
                        robust_ptour(&sample.base_sample.graph, *s, &sample.pred),
                    )];
                    if theoretic {
                        results.push(AlgResult::new(
                            "Robust_pTour_theoretic".into(),
                            theoretic_robust_ptour(&sample.base_sample.graph, *s, &sample.pred),
                        ));
                    }

                    ParamSample::with_results(sample, *s, results)
                })
                .collect::<Vec<ParamSample<'_, PredSample, PredRow, f64>>>()
        })
        .flatten()
        .collect();
    println!("Algorithms with parameters: ✔️");

    export(param_samples, format!("{}.csv", name))
}
