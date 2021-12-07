use std::error::Error;

use graph_explore::{
    algs::{robust_blocking, robust_hdfs, robust_hdfs_rounded, robust_ptour, salesperson},
    graph::Tour,
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    gen_preds::TwoOptPredGen,
    samples::{self, AlgResult, ParamSample, Row, Sample},
    PredConfig,
};

pub fn execute_rosenkrantz_basic(i: usize) -> Result<(), Box<dyn Error>> {
    let mut samples = vec![samples::create_rosen_sample(i)];

    let pred_config = PredConfig {
        num: 10,
        stepsize: 50,
        export: None,
    };

    let pred_samples = samples::create_pred_samples(&mut samples, &pred_config);

    let config = samples::PredParamConfig {
        num: 0,
        start: 0.5,
        step: 0.5,
    };

    let param_samples = samples::create_pred_param_samples(&pred_samples, &config);
    samples::export(param_samples, format!("rosenkrantz_basic_{}.csv", i))
}

pub fn execute_rosenkrantz_robust_lambda(i: usize) -> Result<(), Box<dyn Error>> {
    let samples = vec![samples::create_rosen_sample(i)];
    println!("Base Algorithms: ✔️");

    let params = vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 40.0];

    let param_samples: Vec<ParamSample<'_, Sample, Row, f64>> = samples
        .iter()
        .map(|sample| {
            params
                .par_iter()
                .map(|s| {
                    let results = vec![
                        AlgResult::new("Robust_hDFS".into(), robust_hdfs(&sample.graph, *s)),
                        AlgResult::new(
                            "Robust_Blocking".into(),
                            robust_blocking(&sample.graph, *s, 2.0),
                        ),
                        AlgResult::new(
                            "Robust_hDFS_rounded".into(),
                            robust_hdfs_rounded(&sample.graph, *s),
                        ),
                    ];

                    ParamSample::with_results(sample, *s, results)
                })
                .collect::<Vec<ParamSample<'_, Sample, Row, f64>>>()
        })
        .flatten()
        .collect();

    println!("Robust_hDFS: ✔️");

    samples::export(param_samples, format!("rosenkrantz_robust_{}.csv", i))
}

pub fn execute_rosenkrantz_scaled(i_max: usize) -> Result<(), Box<dyn Error>> {
    let samples: Vec<(usize, Sample)> = (2..i_max + 1)
        .into_par_iter()
        .map(|i| (i, samples::create_rosen_sample(i)))
        .collect();

    let param_samples: Vec<ParamSample<'_, Sample, Row, usize>> = samples
        .par_iter()
        .map(|(i, sample)| {
            let base_tour = &sample.tour;
            let pred: Tour =
                TwoOptPredGen::with_sp_cache(&sample.graph, sample.tour.clone(), 10, &sample.sp)
                    .take_while(|tour| {
                        (tour.cost() - base_tour.cost()).as_float() / sample.mst_cost.as_float()
                            < 5.0
                    })
                    .last()
                    .unwrap();

            let mut results = Vec::<AlgResult>::new();
            let neighbor_tour = pred.to_neighbor_tour(&sample.graph);

            results.push(AlgResult::new(
                format!("pTour (eta/opt = 1)",),
                salesperson(&sample.graph, &neighbor_tour),
            ));
            results.push(AlgResult::new(
                "Robust_pTour (1.0)".into(),
                robust_ptour(&sample.graph, 1.0, &neighbor_tour),
            ));
            results.push(AlgResult::new(
                "Robust_hDFS (1.0)".into(),
                robust_hdfs(&sample.graph, 1.0),
            ));
            results.push(AlgResult::new(
                "Robust_Blocking (1.0)".into(),
                robust_blocking(&sample.graph, 1.0, 2.0),
            ));
            results.push(AlgResult::new(
                "Robust_hDFS_rounded (1.0)".into(),
                robust_hdfs_rounded(&sample.graph, 1.0),
            ));
            results.push(AlgResult::new(
                "Robust_pTour (20.0)".into(),
                robust_ptour(&sample.graph, 20.0, &neighbor_tour),
            ));
            results.push(AlgResult::new(
                "Robust_hDFS (20.0)".into(),
                robust_hdfs(&sample.graph, 20.0),
            ));
            results.push(AlgResult::new(
                "Robust_hDFS_rounded (20.0)".into(),
                robust_hdfs_rounded(&sample.graph, 20.0),
            ));
            results.push(AlgResult::new(
                "Robust_Blocking (20.0)".into(),
                robust_blocking(&sample.graph, 20.0, 2.0),
            ));

            ParamSample::with_results(sample, *i, results)
        })
        .collect();

    samples::export(param_samples, format!("rosenkrantz_scaled_{}.csv", i_max))
}
