use crate::{
    cost::Cost,
    graph::{DetStartNode, Graph, NeighborTour, Node},
    state::State,
};

use super::{Action, DetAlgorithm, Move};

pub fn salesperson<'a, G>(graph: &'a G, tour: &'a NeighborTour) -> Cost
where
    G: Graph<'a> + DetStartNode<'a>,
{
    let start_node = graph.start_node();
    let mut alg = Salesperson::init(graph, start_node, tour);
    alg.explore();
    alg.total_cost()
}

#[derive(Clone, Debug)]
pub struct Salesperson<'a, G>
where
    G: Graph<'a>,
{
    state: State<'a, G>,
    tour: &'a NeighborTour,
    current: usize,
    next_action: Option<Action>,
}

impl<'a, G> Salesperson<'a, G>
where
    G: Graph<'a>,
{
    pub fn init(graph: &'a G, start_node: Node, tour: &'a NeighborTour) -> Self {
        let state = State::<'a>::cached(graph, start_node);

        assert_eq!(tour.get(0), Some(start_node));

        let salesp = Salesperson {
            state,
            tour,
            current: 0,
            next_action: None,
        };
        salesp
    }

    pub fn total_cost(&self) -> Cost {
        self.state.total_cost()
    }

    fn compute_next(&mut self) -> Action {
        self.current += 1;
        while let Some(next) = self.tour.get(self.current) {
            if self.state.explored.contains(&next) {
                self.current += 1;
            } else {
                return Action::Move(Move {
                    current: self.state.current(),
                    cost: self.state.cost_sp_to(next),
                    next,
                });
            }
        }

        return Action::Finished;
    }
}

impl<'a, G> DetAlgorithm<'a, G> for Salesperson<'a, G>
where
    G: Graph<'a> + DetStartNode<'a>,
{
    fn current_node(&self) -> Node {
        self.state.current()
    }

    fn peek_next(&mut self) -> Action {
        if let Some(ref action) = self.next_action {
            action.clone()
        } else {
            self.next_action = Some(self.compute_next());
            self.next_action.clone().unwrap()
        }
    }

    fn explore_next(&mut self) -> Option<Node> {
        if let Action::Move(next_move) = self.peek_next() {
            // move a shortest path to the next vertex
            self.state.move_sp_to(next_move.next);

            self.next_action = None;
            Some(next_move.next)
        } else {
            self.state.back_to_start();
            None
        }
    }
}
