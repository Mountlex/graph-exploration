use graph_explore::{
    graph::{AdjListGraph, GraphSize, Tour},
    sp::ShortestPathsCache,
    two_opt::TwoOpt,
};

pub struct TwoOptPredGen<'a, 'b> {
    graph: &'a AdjListGraph,
    tour: Tour,
    stepsize: usize,
    start: bool,
    sp: &'b ShortestPathsCache,
}

impl<'a, 'b> TwoOptPredGen<'a, 'b> {
    pub fn with_sp_cache(
        graph: &'a AdjListGraph,
        tour: Tour,
        stepsize: usize,
        sp: &'b ShortestPathsCache,
    ) -> Self {
        Self {
            graph,
            tour,
            start: true,
            stepsize,
            sp,
        }
    }
}

impl Iterator for TwoOptPredGen<'_, '_> {
    type Item = Tour;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start {
            self.start = false;
            return Some(self.tour.clone());
        }

        let mut two_opt = TwoOpt::with_sp_cache(
            self.graph.n(),
            self.tour.clone(),
            self.sp,
            graph_explore::two_opt::TwoOptType::Increase,
        );
        let tour = two_opt.run_for(self.stepsize);

        if tour.cost() > self.tour.cost() {
            self.tour = tour;
            Some(self.tour.clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test_predgen {
    use super::*;

    #[test]
    fn test_random_tree_completion() {
        let mut graph = AdjListGraph::new();
        graph.add_edge(1.into(), 2.into(), 1.into());
        graph.add_edge(2.into(), 3.into(), 1.into());
        graph.add_edge(3.into(), 4.into(), 1.into());
    }
}
