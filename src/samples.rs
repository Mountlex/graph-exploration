use std::{
    error::Error,
    fmt::{self, Display, Formatter},
    marker::PhantomData,
    path::Path,
};

use csv::WriterBuilder;
use graph_explore::{
    algs::{robust_ptour, salesperson},
    cost::Cost,
    graph::AdjListGraph,
    graph::Tour,
    graph::{NeighborTour, Tree},
    sp::ShortestPathsCache,
};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressIterator, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;

use crate::{gen_graphs::generate_rosenkrantz, gen_preds::TwoOptPredGen};
use crate::{sample_builder::SampleBuilder, PredConfig};
use crate::{tour_io::export_tour, BaseSampleOptions};

#[derive(Debug, Clone)]
pub struct AlgResult {
    name: String,
    pub(crate) cost: Cost,
}

impl AlgResult {
    pub fn new(name: String, cost: Cost) -> Self {
        Self { name, cost }
    }
}

impl Display for AlgResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.cost)
    }
}

pub trait ResultRow {
    type RowType: Serialize;
    fn to_row(&self) -> Self::RowType;

    fn headers<'a>(&'a self) -> Vec<&'a str>;
}

pub trait ParamExtension<P>: ResultRow + Sized {
    fn with_param<'a>(&'a self, param: P) -> ParamSample<'a, Self, Self::RowType, P>;
}

pub trait IntoParamVec<'a, S, P>
where
    S: ResultRow,
{
    fn into_param_vec(self) -> Vec<ParamSample<'a, S, S::RowType, P>>;
}

impl<'a, P: 'a, S: 'a, I> IntoParamVec<'a, S, P> for I
where
    S: ParamExtension<P>,
    I: Iterator<Item = &'a (P, S)>,
    P: Clone,
{
    fn into_param_vec(self) -> Vec<ParamSample<'a, S, S::RowType, P>> {
        self.map(|(param, sample)| sample.with_param(param.clone()))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub graph: AdjListGraph,
    pub mst: Tree,
    pub tour: Tour,
    pub sp: ShortestPathsCache,
    pub mst_cost: Cost,
    pub results: Vec<AlgResult>,
}

impl<P> ParamExtension<P> for Sample {
    fn with_param<'a>(&'a self, param: P) -> ParamSample<'a, Self, Self::RowType, P> {
        ParamSample::from_base(self, param)
    }
}

#[derive(Serialize, Clone)]
pub struct Row {
    mst_cost: Cost,
    tour_cost: Cost,
    results: Vec<Cost>,
}

impl ResultRow for Sample {
    type RowType = Row;

    fn to_row(&self) -> Self::RowType {
        Row {
            mst_cost: self.mst_cost,
            tour_cost: self.tour.cost(),
            results: self.results.iter().map(|r| r.cost).collect(),
        }
    }

    fn headers<'a>(&'a self) -> Vec<&'a str> {
        let mut header = vec!["mst", "tour"];
        header.append(&mut self.results.iter().map(|res| res.name.as_str()).collect());
        header
    }
}

impl Display for Sample {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "mst: {}, {}",
            self.mst_cost,
            self.results
                .iter()
                .map(|res| format!("{}", res))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[derive(Debug, Clone)]
pub struct PredSample<'a> {
    pub base_sample: &'a Sample,
    pub pred: NeighborTour,
    pub eta: Cost,
    pub results: Vec<AlgResult>,
}

impl<'a> PredSample<'a> {
    pub fn with_results(
        base_sample: &'a Sample,
        pred: NeighborTour,
        results: Vec<AlgResult>,
    ) -> Self {
        PredSample {
            base_sample,
            eta: pred.cost() - base_sample.tour.cost(),
            pred,
            results,
        }
    }
}

#[derive(Serialize, Clone)]
pub struct PredRow {
    base_row: Row,
    eta: Cost,
    results: Vec<Cost>,
}

impl ResultRow for PredSample<'_> {
    type RowType = PredRow;

    fn to_row(&self) -> Self::RowType {
        PredRow {
            base_row: self.base_sample.to_row(),
            eta: self.eta,
            results: self.results.iter().map(|r| r.cost).collect(),
        }
    }

    fn headers<'a>(&'a self) -> Vec<&'a str> {
        let mut header = self.base_sample.headers();
        header.push("eta");
        header.append(&mut self.results.iter().map(|res| res.name.as_str()).collect());
        header
    }
}

#[derive(Clone)]
pub struct ParamSample<'a, S, B, P> {
    base_sample: &'a S,
    param: P,
    results: Vec<AlgResult>,
    phantom_row: PhantomData<B>,
}

impl<'a, S, B, P> ParamSample<'a, S, B, P> {
    pub fn from_base(base_sample: &'a S, param: P) -> Self {
        ParamSample {
            base_sample,
            param,
            results: vec![],
            phantom_row: PhantomData,
        }
    }

    pub fn with_results(base_sample: &'a S, param: P, results: Vec<AlgResult>) -> Self {
        ParamSample {
            base_sample,
            param,
            results,
            phantom_row: PhantomData,
        }
    }
}

#[derive(Serialize, Clone)]
pub struct ParamRow<B, P>
where
    B: Serialize,
    P: Serialize,
{
    base_row: B,
    param: P,
    results: Vec<Cost>,
}

impl<S, B, P> ResultRow for ParamSample<'_, S, B, P>
where
    P: Serialize + Clone,
    B: Serialize,
    S: ResultRow + ResultRow<RowType = B>,
{
    type RowType = ParamRow<B, P>;

    fn to_row(&self) -> Self::RowType {
        ParamRow {
            base_row: self.base_sample.to_row(),
            param: self.param.clone(),
            results: self.results.iter().map(|r| r.cost).collect(),
        }
    }

    fn headers<'a>(&'a self) -> Vec<&'a str> {
        let mut header = self.base_sample.headers();
        header.push("param");
        header.append(&mut self.results.iter().map(|res| res.name.as_str()).collect());
        header
    }
}

pub fn create_rosen_sample(i: usize) -> Sample {
    let (graph, tour) = generate_rosenkrantz(i);
    SampleBuilder::new()
        .with_tour(tour)
        .with_base_algorithms()
        .build(graph)
}

pub fn create_samples(
    graphs: Vec<AdjListGraph>,
    option: BaseSampleOptions,
) -> Result<Vec<Sample>, Box<dyn Error>> {
    let num = graphs.len() as u64;
    let pb = ProgressBar::new(num);
    pb.set_style(
        ProgressStyle::default_bar().template(
            "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len})",
        ),
    );
    pb.enable_steady_tick(20);
    pb.set_message("Base Samples");

    let samples: Vec<Sample> = graphs
        .into_par_iter()
        .progress_with(pb)
        .map(|graph| {
            SampleBuilder::new()
                .with_imported_tour_or_default(option.read_tour.clone())
                .with_export_tour(option.write_tour.clone())
                .with_base_algorithms()
                .with_two_opt(option.two_opt)
                .build(graph)
        })
        .collect();

    log::info!("Finished evaluating base algorithms.");

    println!("Base algorithms: ✔️");
    Ok(samples)
}

pub fn create_pred_samples<'a>(samples: &'a [Sample], config: &PredConfig) -> Vec<PredSample<'a>> {
    let num = samples.len() as u64;
    let pb = ProgressBar::new(num);
    pb.set_style(
        ProgressStyle::default_bar().template(
            "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len})",
        ),
    );
    pb.enable_steady_tick(20);
    pb.set_message("Predictions");

    let res = samples
        .iter()
        .progress_with(pb)
        .flat_map(|sample| {
            let preds: Vec<Tour> = TwoOptPredGen::with_sp_cache(
                &sample.graph,
                sample.tour.clone(),
                config.stepsize,
                &sample.sp,
            )
            .take(config.num)
            .collect::<Vec<Tour>>();
            log::info!("Generated {} predictions.", preds.len());
            log::info!("Starting to evaluate prediction algorithms");
            if let Some(dir) = &config.export {
                std::fs::create_dir_all(&dir).unwrap();
                if dir.is_dir() {
                    for (i, tour) in preds.iter().enumerate() {
                        let mut file = dir.clone();
                        file.push(format!("{}.txt", i));
                        export_tour(&file, tour).unwrap();
                    }
                }
            }
            let pred_samples = preds
                .into_par_iter()
                .map(|tour| {
                    let neighbor_tour = tour.to_neighbor_tour(&sample.graph);
                    let results = simulate_pred_algorithms(&sample.graph, &neighbor_tour);
                    let eta = tour.cost() - sample.tour.cost();
                    if eta == 0.into() {
                        // assert_eq!(results.first().unwrap().cost, tour.cost());
                    }
                    PredSample {
                        base_sample: sample,
                        eta,
                        pred: neighbor_tour,
                        results,
                    }
                })
                .collect::<Vec<PredSample<'a>>>();

            let max_rel_error = pred_samples
                .iter()
                .map(|s| s.eta.as_float() / s.base_sample.mst_cost.as_float())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            log::info!("Max relative error: {}", max_rel_error);
            pred_samples
        })
        .collect();
    log::info!("Finished evaluating prediction algorithms");

    println!("Predictions: ✔️");
    res
}

fn simulate_pred_algorithms(graph: &AdjListGraph, pred: &NeighborTour) -> Vec<AlgResult> {
    let mut results = vec![];
    results.push(AlgResult {
        name: "pTour".into(),
        cost: salesperson(graph, &pred),
    });
    // results.push(AlgResult {
    //     name: "pBlocking".into(),
    //     cost: pblocking(graph, pred),
    // });
    results
}

type PredParamSample<'a, P> = ParamSample<'a, PredSample<'a>, PredRow, P>;

pub struct PredParamConfig {
    pub num: usize,
    pub start: f64,
    pub step: f64,
}

pub fn create_pred_param_samples<'a>(
    samples: &'a [PredSample<'a>],
    config: &PredParamConfig,
) -> Vec<PredParamSample<'a, f64>> {
    let num = samples.len() as u64;
    let pb = ProgressBar::new(num);
    pb.set_style(
        ProgressStyle::default_bar().template(
            "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len})",
        ),
    );
    pb.enable_steady_tick(20);
    pb.set_message("Parameterized algorithms");

    log::info!("Starting to evaluate parameterized algorithms.");

    let res = samples
        .par_iter()
        .progress_with(pb)
        .flat_map(|sample| {
            (0..config.num)
                .map(|i| {
                    let param = config.start + (config.step * i as f64);
                    let results =
                        simulate_param_algorithms(&sample.base_sample.graph, &sample.pred, param);
                    PredParamSample::with_results(sample, param, results)
                })
                .collect::<Vec<PredParamSample<f64>>>()
        })
        .collect();
    println!("Parameterized algorithms: ✔️");
    res
}

fn simulate_param_algorithms(
    graph: &AdjListGraph,
    pred: &NeighborTour,
    param: f64,
) -> Vec<AlgResult> {
    let mut results = vec![];

    results.push(AlgResult {
        name: "Robust_pTour".into(),
        cost: robust_ptour(graph, param, pred),
    });

    results
}

pub fn export<I: ResultRow, P: AsRef<Path>>(
    samples: Vec<I>,
    path: P,
) -> Result<(), Box<dyn Error>> {
    log::info!("Exporting results to {:?}.", path.as_ref());
    let headers = samples.first().unwrap().headers();
    let mut wtr = WriterBuilder::new().has_headers(false).from_path(path)?;
    wtr.write_record(headers)?;
    for sample in samples {
        let row = sample.to_row();
        wtr.serialize(row)?;
    }
    wtr.flush()?;
    Ok(())
}
