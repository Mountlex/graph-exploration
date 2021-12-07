use std::ops::Index;

use super::Node;

#[derive(Debug, Clone)]
pub struct NodeIndex {
    index: Vec<usize>,
    min_node_index: usize,
}

impl NodeIndex {
    pub fn init(nodes: &[Node]) -> Self {
        let min_node_index = (&nodes).into_iter().map(|node| node.id()).min().unwrap();
        let max_node_index = (&nodes).into_iter().map(|node| node.id()).max().unwrap();

        let mut index: Vec<usize> = vec![0; max_node_index - min_node_index + 1];
        for (i, node) in nodes.into_iter().enumerate() {
            index[node.id() - min_node_index] = i;
        }

        Self {
            index,
            min_node_index,
        }
    }

    pub fn empty() -> Self {
        Self {
            index: vec![],
            min_node_index: 0,
        }
    }

    pub fn get(&self, node: &Node) -> Option<usize> {
        self.index.get(node.id() - self.min_node_index).copied()
    }
}

impl Index<&Node> for NodeIndex {
    type Output = usize;

    fn index(&self, node: &Node) -> &Self::Output {
        &self.index[node.id() - self.min_node_index]
    }
}
