use std::error::Error;

use graph_explore::{
    algs::{robust_blocking, robust_ptour, salesperson},
    graph::Tour,
};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    gen_graphs::generate_blocking_lb,
    gen_preds::TwoOptPredGen,
    sample_builder::SampleBuilder,
    samples::{self, AlgResult, ParamSample, Row, Sample},
};

pub fn execute_blocking_lb_robust(i: usize, delta: f32) -> Result<(), Box<dyn Error>> {
    let (graph, tour) = generate_blocking_lb(i, delta);
    let sample = SampleBuilder::new()
        .with_tour(tour)
        .with_base_algorithms()
        .build(graph);
    println!("Base Algorithms: ✔️");

    let params = vec![0.0, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0];

    let param_samples: Vec<ParamSample<'_, Sample, Row, f64>> = params
        .par_iter()
        .map(|s| {
            let results = vec![AlgResult::new(
                "Robust_Blocking".into(),
                robust_blocking(&sample.graph, *s, delta),
            )];

            ParamSample::with_results(&sample, *s, results)
        })
        .collect::<Vec<ParamSample<'_, Sample, Row, f64>>>();

    println!("Robust_Blocking: ✔️");

    samples::export(param_samples, format!("blocking_robust_{}.csv", i))
}

pub fn execute_blocking_lb_scaled(i_max: usize, delta: f32) -> Result<(), Box<dyn Error>> {
    let samples: Vec<(usize, Sample)> = (2..i_max + 1)
        .into_par_iter()
        .map(|i| {
            let (graph, tour) = generate_blocking_lb(i, delta);
            let sample = SampleBuilder::new()
                .with_tour(tour)
                .with_base_algorithms()
                .build(graph);
            (i, sample)
        })
        .collect();
    println!("Base Algorithms: ✔️");
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
                format!("pTour (eta/opt = 5)",),
                salesperson(&sample.graph, &neighbor_tour),
            ));
            results.push(AlgResult::new(
                "Robust_pTour (1.0)".into(),
                robust_ptour(&sample.graph, 1.0, &neighbor_tour),
            ));
            results.push(AlgResult::new(
                "Robust_Blocking (1.0)".into(),
                robust_blocking(&sample.graph, 1.0, delta),
            ));

            results.push(AlgResult::new(
                "Robust_pTour (20.0)".into(),
                robust_ptour::<_>(&sample.graph, 20.0, &neighbor_tour),
            ));
            results.push(AlgResult::new(
                "Robust_Blocking (20.0)".into(),
                robust_blocking(&sample.graph, 20.0, delta),
            ));

            ParamSample::with_results(sample, *i, results)
        })
        .collect();
    samples::export(param_samples, format!("blocking_scaled_{}.csv", i_max))
}
