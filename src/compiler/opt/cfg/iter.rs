use std::collections::HashSet;

use crate::compiler::opt::Block;

/// Post-order, depth-first iterator, over *reachable* basic blocks in a
/// control-flow graph.
#[derive(Debug)]
pub struct PostOrder<'a> {
    blocks: Vec<&'a Block<'a>>,
    index: usize,
}

impl<'a> PostOrder<'a> {
    /// Creates a new `PostOrder` iterator over *reachable* basic blocks.
    ///
    /// A block is considered reachable if it is a successor of any prior block.
    #[must_use]
    fn new(blocks: &'a [Block<'a>]) -> Self {
        let mut post_order = Self {
            blocks: vec![],
            index: 0,
        };

        if let Some(Block::Entry { successor }) = blocks.first() {
            post_order.iterative_post_order(blocks, *successor);
        }

        debug_assert!(
            !post_order.blocks.is_empty(),
            "malformed control-flow graph: missing entry block"
        );

        post_order
    }

    /// Performs an iterative depth-first search on the blocks, pushing blocks
    /// onto `self` in post-order.
    fn iterative_post_order(&mut self, blocks: &'a [Block<'a>], entry: usize) {
        let mut visited = HashSet::new();
        let mut stack = vec![(entry, false)];

        while let Some((id, children_done)) = stack.pop() {
            if children_done && let Some(block) = blocks.iter().find(|b| b.id() == id) {
                self.blocks.push(block);
            } else if let Some(Block::Basic { successors, .. }) =
                blocks.iter().find(|b| b.id() == id)
            {
                stack.push((id, true));

                for &succ_id in successors.iter().rev() {
                    if visited.insert(succ_id) {
                        stack.push((succ_id, false));
                    }
                }
            }
        }
    }
}

impl<'a> Iterator for PostOrder<'a> {
    type Item = &'a Block<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.blocks.get(self.index)?;
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
