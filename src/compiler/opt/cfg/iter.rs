use std::collections::HashSet;

use crate::compiler::opt::Block;

/// Post-order, depth-first iterator, over *reachable* basic blocks in a
/// control-flow graph.
#[derive(Debug)]
pub struct PostOrder<'a> {
    post: Vec<&'a Block<'a>>,
    index: usize,
}

impl<'a> PostOrder<'a> {
    /// Creates a new `PostOrder` iterator over *reachable* basic blocks.
    ///
    /// A block is considered reachable if it is a successor of any prior block.
    #[must_use]
    fn new(blocks: &'a [Block<'a>]) -> Self {
        // TODO: Make the traversal iterative.
        let mut post_order = Self {
            post: vec![],
            index: 0,
        };

        let mut visited = HashSet::new();

        if let Some(Block::Entry { successor }) = blocks.first() {
            post_order.recursive_dfs(blocks, *successor, &mut visited);
        }

        post_order
    }

    /// Performs a depth-first search on the blocks, pushing them onto `self` in
    /// post-order.
    fn recursive_dfs(&mut self, blocks: &'a [Block<'a>], id: usize, visited: &mut HashSet<usize>) {
        // NOTE: O(n) time complexity.
        let block = &blocks
            .iter()
            .find(|block| block.id() == id)
            .expect("all blocks should be present during traversal");

        // Only collect basic blocks during traversal.
        if let Block::Basic { successors, .. } = block
            && visited.insert(id)
        {
            for block_id in successors {
                self.recursive_dfs(blocks, *block_id, visited);
            }

            self.post.push(block);
        }
    }
}

impl<'a> Iterator for PostOrder<'a> {
    type Item = &'a Block<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.post.get(self.index)?;
        self.index += 1;
        Some(result)
    }
}

/// Immutable slice over the blocks of a control-flow graph (excluding entry
/// and exit blocks).
#[derive(Debug)]
pub struct BasicBlocks<'a> {
    blocks: &'a [Block<'a>],
}

impl<'a> BasicBlocks<'a> {
    /// Returns a new `BasicBlocks`.
    #[inline]
    #[must_use]
    pub const fn new(blocks: &'a [Block<'a>]) -> Self {
        Self { blocks }
    }

    /// Returns an iterator over basic blocks.
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'a, Block<'a>> {
        let exit_block_idx = self.blocks.len() - 1;
        self.blocks[Block::ENTRY_ID + 1..exit_block_idx].iter()
    }

    /// Returns a post-order, depth-first iterator, over the *reachable* basic
    /// blocks.
    #[inline]
    #[must_use]
    pub fn post_order(&self) -> PostOrder<'a> {
        PostOrder::new(self.blocks)
    }

    /// Returns the number of basic blocks.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        // Don't include entry or exit block in count.
        self.blocks.len() - 2
    }

    /// Returns `true` if the slice contains no basic blocks.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        // Only contains entry and exit blocks.
        self.blocks.len() == 2
    }
}

impl<'a> IntoIterator for &BasicBlocks<'a> {
    type Item = &'a Block<'a>;
    type IntoIter = std::slice::Iter<'a, Block<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.iter()
    }
}
