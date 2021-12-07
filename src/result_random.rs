use std::error::Error;

use graph_explore::algs::robust_hdfs;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    gen_graphs::RandomGraphGenerator,
    sample_builder::SampleBuilder,
    samples::{self, AlgResult, IntoParamVec, ParamSample, Row, Sample},
};

pub fn execute_random_graphs_basic(n_max: usize, step: usize) -> Result<(), Box<dyn Error>> {
    let mut graphs = vec![];
    let mut n = step;
    while n <= n_max {
        graphs.push((
            n,
            RandomGraphGenerator::default_costs(n, 0.5).next().unwrap(),
        ));
        n += step;
    }

    let samples: Vec<(usize, Sample)> = graphs
        .into_par_iter()
        .map(|(n, graph)| {
            let sample = SampleBuilder::new()
                .with_base_algorithms()
                .with_results(AlgResult::new(
                    "Robust_hDFS (1.0)".into(),
                    robust_hdfs(&graph, 1.0),
                ))
                .with_results(AlgResult::new(
                    "Robust_hDFS (10.0)".into(),
                    robust_hdfs(&graph, 10.0),
                ))
                .build(graph);

            (n, sample)
        })
        .collect();

    let param_samples: Vec<ParamSample<'_, Sample, Row, usize>> = samples.iter().into_param_vec();
    samples::export(
        param_samples,
        format!("random_basic_{}_{}.csv", step, n_max),
    )
}
