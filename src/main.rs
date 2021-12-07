pub mod gen_graphs;
mod gen_preds;
mod graphml_import;
mod pbf_import;
mod sample_builder;
mod samples;
mod tour_io;
mod tsplib_import;

mod result_blocking_lb;
mod result_cities;
mod result_random;
mod result_rosenkrantz;
mod result_tsplib;

use graphml_import::graphml_import;
use indicatif::{ProgressBar, ProgressIterator, ProgressStyle};
use result_blocking_lb::{execute_blocking_lb_robust, execute_blocking_lb_scaled};
use result_cities::execute_city;
use result_random::execute_random_graphs_basic;
use result_rosenkrantz::{
    execute_rosenkrantz_basic, execute_rosenkrantz_robust_lambda, execute_rosenkrantz_scaled,
};
use result_tsplib::execute_tsplib_experiment;
use std::{error::Error, path::PathBuf};

use clap::Clap;
use gen_graphs::RandomGraphGenerator;
use graph_explore::{
    graph::{AdjListGraph, Connected, GraphSize},
    tsp::MatchingAlgorithm,
};
use pbf_import::pbf_import;
use samples::{
    create_pred_param_samples, create_pred_samples, create_samples, export, ParamExtension,
    ParamSample, PredParamConfig, Row, Sample,
};
use tsplib_import::xml_import;

use crate::samples::create_rosen_sample;

pub use gen_graphs::generate_rosenkrantz;

#[derive(Clap)]
enum Cli {
    Import(Import),
    Gen(Gen),
    Rosen(Rosenkrantz),
    DetDensity(DetDensity),
    Tsplib,
    Rosenbasic(Rosenbasic),
    Rosenrobust(Rosenrobust),
    Rosenscaled(Rosenscaled),
    Randombasic(Randombasic),
    BlockingRobust(BlockingLB),
    BlockingScaled(BlockingLBScaling),
    City(City),
}

#[derive(Clap)]
struct BlockingLB {
    i: usize,
    delta: f32,
}

#[derive(Clap)]
struct City {
    name: String,

    #[clap(short, long)]
    theoretic: bool,
}

#[derive(Clap)]
struct BlockingLBScaling {
    i_max: usize,
    delta: f32,
}

#[derive(Clap)]
struct Rosenbasic {
    i: usize,
}

#[derive(Clap)]
struct Randombasic {
    step: usize,
    n_max: usize,
}

#[derive(Clap)]
struct Rosenrobust {
    i: usize,
}

#[derive(Clap)]
struct Rosenscaled {
    i_max: usize,
}

#[derive(Clap)]
struct Import {
    #[clap(short, long)]
    max_nodes: Option<usize>,

    #[clap(short, long)]
    num_graphs: Option<usize>,

    /// Import
    #[clap(parse(from_os_str))]
    input: PathBuf,

    #[clap(flatten)]
    options: BaseSampleOptions,

    #[clap(
        short,
        long,
        global = true,
        default_value = "results.csv",
        parse(from_os_str)
    )]
    output: PathBuf,

    #[clap(subcommand)]
    pred_gen: Option<PredGen>,
}

#[derive(Clap, Clone, Debug, Default)]
pub struct BaseSampleOptions {
    #[clap(long, default_value = "blossom")]
    matching: MatchingAlgorithm,
    /// Import tour
    #[clap(short, long, parse(from_os_str))]
    read_tour: Option<PathBuf>,

    /// Import tour
    #[clap(long)]
    two_opt: Option<usize>,

    /// Import tour
    #[clap(short, long, parse(from_os_str))]
    write_tour: Option<PathBuf>,
}

#[derive(Clap)]
struct Gen {
    #[clap(short, long, default_value = "50")]
    num_nodes: usize,

    #[clap(short, long, default_value = "0.5")]
    density: f64,

    #[clap()]
    num: usize,

    #[clap(flatten)]
    options: BaseSampleOptions,

    #[clap(
        short,
        long,
        global = true,
        default_value = "results.csv",
        parse(from_os_str)
    )]
    output: PathBuf,

    #[clap(subcommand)]
    pred_gen: Option<PredGen>,
}

#[derive(Clap)]
enum PredGen {
    Preds(Preds),
}

#[derive(Clap)]
struct Preds {
    #[clap(flatten)]
    config: PredConfig,

    #[clap(subcommand)]
    param_gen: Option<ParamGen>,
}

#[derive(Clap, Clone, Debug)]
pub struct PredConfig {
    pub num: usize,

    #[clap(short, long, default_value = "50")]
    pub stepsize: usize,

    #[clap(long, parse(from_os_str))]
    pub export: Option<PathBuf>,
}

#[derive(Clap)]
enum ParamGen {
    Lin(LinParams),
}
#[derive(Clap)]
struct LinParams {
    #[clap(short, long, default_value = "5")]
    num: usize,

    #[clap(long, default_value = "0.3")]
    start: f64,

    #[clap(long, default_value = "0.3")]
    step: f64,
}

#[derive(Clap)]
struct DetDensity {
    #[clap(short, long, default_value = "50")]
    num_nodes: usize,

    #[clap(short, long, default_value = "0.2")]
    density_stepsize: f64,

    #[clap()]
    num: usize,

    #[clap(flatten)]
    options: BaseSampleOptions,

    #[clap(short, long, default_value = "results.csv", parse(from_os_str))]
    output: PathBuf,
}

#[derive(Clap)]
struct Rosenkrantz {
    i: usize,

    #[clap(short, long, default_value = "results.csv", parse(from_os_str))]
    output: PathBuf,

    #[clap(subcommand)]
    pred_gen: Option<PredGen>,
}

fn import_from_file(file: PathBuf, max_nodes: Option<usize>) -> Option<AdjListGraph> {
    if file.extension().unwrap() == "xml" {
        return Some(xml_import(file, max_nodes));
    } else if file.extension().unwrap() == "pdf" {
        return Some(pbf_import(file));
    } else if file.extension().unwrap() == "graphml" {
        return Some(graphml_import(file, max_nodes));
    }

    None
}

fn evaluate_graphs(
    pred_gen: Option<PredGen>,
    graphs: Vec<AdjListGraph>,
    sample_options: BaseSampleOptions,
    output: PathBuf,
) -> Result<(), Box<dyn Error>> {
    let samples = create_samples(graphs, sample_options)?;
    evaluate_samples(pred_gen, samples, output)
}

fn evaluate_samples(
    pred_gen: Option<PredGen>,
    mut samples: Vec<Sample>,
    output: PathBuf,
) -> Result<(), Box<dyn Error>> {
    if let Some(predgen) = pred_gen {
        match predgen {
            PredGen::Preds(preds) => {
                let pred_samples = create_pred_samples(&mut samples, &preds.config);
                if let Some(paramgen) = preds.param_gen {
                    match paramgen {
                        ParamGen::Lin(lin_gen) => {
                            let config = PredParamConfig {
                                num: lin_gen.num,
                                start: lin_gen.start,
                                step: lin_gen.step,
                            };
                            let param_samples = create_pred_param_samples(&pred_samples, &config);
                            export(param_samples, output)?;
                        }
                    }
                } else {
                    export(pred_samples, output)?;
                }
            }
        }
    } else {
        export(samples, output)?;
    }
    Ok(())
}

fn set_up_logging() -> Result<(), fern::InitError> {
    std::fs::create_dir_all("logs")?;
    fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "[{date}][{level}] {message}",
                date = chrono::Local::now().format("%H:%M:%S"),
                level = record.level(),
                message = message
            ));
        })
        .level(log::LevelFilter::Info)
        .chain(fern::log_file(format!(
            "logs/{}.log",
            chrono::Local::now().format("%d%m%Y-%H%M")
        ))?)
        .apply()?;

    log::info!("Logger set up!");

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    set_up_logging()?;
    let cli = Cli::parse();

    match cli {
        Cli::Import(import) => {
            let mut graphs: Vec<AdjListGraph> = vec![];
            if import.input.is_file() {
                if let Some(graph) = import_from_file(import.input, import.max_nodes) {
                    println!(
                        "Graph with {} nodes and {} edges imported.",
                        graph.n(),
                        graph.m()
                    );
                    graphs.push(graph);
                }
            } else {
                let paths: Vec<std::fs::DirEntry> = std::fs::read_dir(import.input.clone())?
                    .filter_map(|e| e.ok())
                    .collect();
                let num = paths.len().min(import.num_graphs.unwrap_or(usize::MAX));

                let pb = ProgressBar::new(num as u64);
                pb.set_style(ProgressStyle::default_bar().template(
                    "{msg} {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len})",
                ));
                pb.enable_steady_tick(20);
                pb.set_message("Import");
                graphs = paths
                    .iter()
                    .progress_with(pb)
                    .take(num)
                    .flat_map(|path| import_from_file(path.path(), import.max_nodes))
                    .collect();

                graphs.iter().for_each(|g| assert!(g.connected()))
            }
            println!("Imported {} graphs.", graphs.len());
            if !graphs.is_empty() {
                evaluate_graphs(import.pred_gen, graphs, import.options, import.output)?;
            }
        }
        Cli::Gen(gen) => {
            println!("{:?}", gen.output);
            let graphs: Vec<AdjListGraph> =
                RandomGraphGenerator::default_costs(gen.num_nodes, gen.density)
                    .take(gen.num)
                    .collect();

            evaluate_graphs(gen.pred_gen, graphs, gen.options, gen.output)?;
        }
        Cli::Rosen(rosen) => {
            println!("{:?}", rosen.output);
            let output = rosen.output;
            let samples: Vec<Sample> = vec![create_rosen_sample(rosen.i)];
            evaluate_samples(rosen.pred_gen, samples, output)?;
        }
        Cli::DetDensity(gen) => {
            let mut dense = 0.0;
            let mut samples: Vec<(f64, Vec<Sample>)> = vec![];

            while dense <= 1.0 {
                let graphs: Vec<AdjListGraph> =
                    RandomGraphGenerator::default_costs(gen.num_nodes, dense)
                        .take(gen.num)
                        .collect();
                let s = create_samples(graphs, gen.options.clone())?;
                samples.push((dense, s));
                dense += gen.density_stepsize;
            }

            let param_samples: Vec<ParamSample<'_, Sample, Row, f64>> = samples
                .iter()
                .map(|(d, s)| {
                    s.iter()
                        .map(|s| s.with_param(*d))
                        .collect::<Vec<ParamSample<'_, Sample, Row, f64>>>()
                })
                .flatten()
                .collect();

            export(param_samples, gen.output)?;
        }
        Cli::Tsplib => execute_tsplib_experiment()?,
        Cli::Rosenbasic(r) => execute_rosenkrantz_basic(r.i)?,
        Cli::Rosenrobust(r) => execute_rosenkrantz_robust_lambda(r.i)?,
        Cli::Rosenscaled(r) => execute_rosenkrantz_scaled(r.i_max)?,
        Cli::Randombasic(r) => execute_random_graphs_basic(r.n_max, r.step)?,
        Cli::BlockingRobust(b) => execute_blocking_lb_robust(b.i, b.delta)?,
        Cli::BlockingScaled(b) => execute_blocking_lb_scaled(b.i_max, b.delta)?,
        Cli::City(city) => execute_city(city.name, city.theoretic)?,
    }
    Ok(())
}
