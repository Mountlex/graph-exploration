use std::error::Error;

use graph_explore::{
    algs::{robust_blocking, robust_hdfs, robust_hdfs_rounded},
    graph::AdjListGraph,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    import_from_file,
    sample_builder::SampleBuilder,
    samples::{export, AlgResult, ParamSample, Row, Sample},
};

pub fn execute_tsplib_experiment() -> Result<(), Box<dyn Error>> {
    let paths: Vec<std::fs::DirEntry> = std::fs::read_dir("resources/sym_tsplib/set1")?
        .filter_map(|e| e.ok())
        .collect();

    let graphs: Vec<AdjListGraph> = paths
        .into_iter()
        .flat_map(|path| import_from_file(path.path(), None))
        .collect();

    let samples: Vec<Sample> = graphs
        .into_iter()
        .map(|graph| {
            SampleBuilder::new()
                .with_christofides_tour_default()
                .with_base_algorithms()
                .build(graph)
        })
        .collect();

    let params = vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 40.0];
    let param_samples: Vec<ParamSample<'_, Sample, Row, f64>> = samples
        .par_iter()
        .map(|sample| {
            params
                .iter()
                .map(|s| {
                    let results = vec![
                        AlgResult::new("Robust_hDFS".into(), robust_hdfs(&sample.graph, *s)),
                        AlgResult::new(
                            "Robust_hDFS_rounded".into(),
                            robust_hdfs_rounded(&sample.graph, *s),
                        ),
                        AlgResult::new(
                            "Robust_Blocking".into(),
                            robust_blocking(&sample.graph, *s, 2.0),
                        ),
                    ];

                    ParamSample::with_results(sample, *s, results)
                })
                .collect::<Vec<ParamSample<'_, Sample, Row, f64>>>()
        })
        .flatten()
        .collect();

    println!("Robust: ✔️");
    export(param_samples, "tsplib_robust.csv")?;

    Ok(())
}
