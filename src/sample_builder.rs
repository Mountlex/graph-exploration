use std::path::PathBuf;

use graph_explore::{
    algs::{blocking, dfs, hdfs_rounding, nearest_neighbor},
    graph::{AdjListGraph, Tour},
    mst::prims_tree,
    sp::ShortestPathsCache,
    tsp::{christofides, MatchingAlgorithm},
};

use crate::{
    samples::{AlgResult, Sample},
    tour_io::{check_tour, export_tour, import_tour},
};

#[derive(Debug, Clone)]
pub struct SampleBuilder {
    compute_tour: bool,
    read_tour: Option<PathBuf>,
    write_tour: Option<PathBuf>,
    tour: Option<Tour>,
    two_opt_iterations: Option<usize>,
    matching: MatchingAlgorithm,
    results: Vec<AlgResult>,
    blocking: bool,
    dfs: bool,
    hdfs: bool,
    nn: bool,
}

impl SampleBuilder {
    pub fn new() -> Self {
        SampleBuilder {
            compute_tour: false,
            two_opt_iterations: None,
            read_tour: None,
            tour: None,
            write_tour: None,
            matching: MatchingAlgorithm::Blossom,
            blocking: false,
            results: vec![],
            dfs: false,
            hdfs: false,
            nn: false,
        }
    }

    pub fn build(mut self, graph: AdjListGraph) -> Sample {
        let (mst, mst_cost) = prims_tree(&graph);
        let sp = ShortestPathsCache::compute_all_pairs_par(&graph);

        log::info!("Simulating base algorithms.");
        let mut res_hdfs: Option<AlgResult> = None;
        let mut res_dfs: Option<AlgResult> = None;
        let mut res_blocking: Option<AlgResult> = None;
        let mut res_nn: Option<AlgResult> = None;
        rayon::scope(|s| {
            if self.hdfs {
                s.spawn(|_| res_hdfs = Some(AlgResult::new("hDFS".into(), hdfs_rounding(&graph))))
            };
            if self.dfs {
                s.spawn(|_| res_dfs = Some(AlgResult::new("DFS".into(), dfs(&graph))));
            }
            if self.nn {
                s.spawn(|_| res_nn = Some(AlgResult::new("NN".into(), nearest_neighbor(&graph))));
            }
            if self.blocking {
                s.spawn(|_| {
                    res_blocking = Some(AlgResult::new("Blocking".into(), blocking(&graph)))
                });
            }
        });

        if self.blocking {
            self.results.push(res_blocking.unwrap());
        }
        if self.hdfs {
            self.results.push(res_hdfs.unwrap());
        }
        if self.nn {
            self.results.push(res_nn.unwrap());
        }
        if self.dfs {
            self.results.push(res_dfs.unwrap());
        }
        log::info!("Finished base algorithms.");

        let mut tour: Option<Tour> = self.tour;
        if let Some(file) = &self.read_tour {
            log::info!("Imported tsp tour from {:?}.", file.file_name().unwrap());
            let imported_tour = import_tour(file, &sp).unwrap();
            if check_tour(&graph, &imported_tour) {
                log::info!(
                    "Tour valid: using tsp tour from {:?}.",
                    file.file_name().unwrap()
                );
                tour = Some(imported_tour);
            }
        }
        if tour.is_none() && self.compute_tour {
            tour = Some(christofides(
                &graph,
                &mst,
                &sp,
                self.matching,
                self.two_opt_iterations,
            ));
            if let Some(write) = &self.write_tour {
                log::info!("Exporting tsp tour to {:?}.", write.file_name().unwrap());
                export_tour(write, tour.as_ref().unwrap()).unwrap();
            }
        }

        Sample {
            graph,
            mst,
            tour: tour.unwrap_or_else(|| Tour::empty()),
            sp,
            mst_cost,
            results: self.results,
        }
    }

    pub fn with_imported_tour_or_default(mut self, filename: Option<PathBuf>) -> Self {
        self.compute_tour = true;
        self.read_tour = filename;
        self.matching = MatchingAlgorithm::Blossom;
        self.two_opt_iterations = None;
        self
    }

    pub fn with_two_opt(mut self, iterations: Option<usize>) -> Self {
        self.two_opt_iterations = iterations;
        self
    }

    pub fn with_export_tour(mut self, filename: Option<PathBuf>) -> Self {
        self.write_tour = filename;
        self
    }

    pub fn with_tour(mut self, tour: Tour) -> Self {
        self.tour = Some(tour);
        self
    }

    pub fn with_christofides_tour_default(mut self) -> Self {
        self.compute_tour = true;
        self.matching = MatchingAlgorithm::Blossom;
        self.two_opt_iterations = None;
        self
    }

    pub fn with_blocking(mut self) -> Self {
        self.blocking = true;
        self
    }

    pub fn with_hdfs(mut self) -> Self {
        self.hdfs = true;
        self
    }

    pub fn with_nn(mut self) -> Self {
        self.nn = true;
        self
    }

    pub fn with_dfs(mut self) -> Self {
        self.dfs = true;
        self
    }

    pub fn with_base_algorithms(self) -> Self {
        self.with_blocking().with_dfs().with_hdfs().with_nn()
    }

    pub fn with_results(mut self, result: AlgResult) -> Self {
        self.results.push(result);
        self
    }
}
