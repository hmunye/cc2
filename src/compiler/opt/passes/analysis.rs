//! Data Flow Analysis
//!
//! Generic framework for bi-directional data-flow analyses over a control-flow
//! graph (_CFG_).

use std::collections::{HashSet, VecDeque};

use crate::compiler::opt::{Block, CFG};

/// Trait for performing data-flow analysis over a control-flow graph
/// (bi-directional).
pub trait DataFlowAnalysis<'a> {
    /// Information being tracked during analysis.
    type Fact: Clone + PartialEq;

    /// Propagates the incoming fact through the block's instructions, updating
    /// the fact for each instruction and the block's exit.
    fn transfer(&mut self, block: &Block<'a>, incoming: Self::Fact);

    /// Merges facts from multiple execution paths (predecessors or successors),
    /// starting from the given block, returning the intersection of the initial
    /// fact and the facts from neighboring blocks.
    fn meet(&self, block: &Block<'_>, initial: &Self::Fact) -> Self::Fact;

    /// Returns the initial facts for analysis (e.g., identity element for
    /// `meet` function).
    fn initial(&self, cfg: &'a CFG<'a>) -> Self::Fact;

    /// Stores the fact for the block identified by `id`, replacing any existing
    /// fact.
    fn record_block_fact(&mut self, block_id: usize, num_inst: usize, fact: &Self::Fact);

    /// Returns the recorded fact for the block identified by `id`, or `None` if
    /// no fact has been stored for the block.
    fn get_block_fact(&self, block_id: usize) -> Option<&Self::Fact>;

    /// Returns `true` if this is a forward analysis.
    fn is_forward(&self) -> bool;
}

/// Fixed-point solver for a data-flow analysis over a control-flow graph, which
/// iterates over each block until facts converge.
///
/// # Panics
///
/// Panics if the control-flow graph is malformed.
pub fn run_analysis<'a, A: DataFlowAnalysis<'a>>(cfg: &'a CFG<'a>, a: &mut A) {
    let init = a.initial(cfg);

    // Exclude the entry and exit block IDs from the set.
    let mut seen_blocks = HashSet::with_capacity(cfg.blocks.len() - 2);

    let mut worklist: VecDeque<_> = cfg
        .basic_blocks()
        // Working in post-order minimizes the number of times a block needs
        // to be revisited for analysis.
        .post_order()
        .inspect(|block| {
            let id = block.id();
            seen_blocks.insert(id);
            if let Block::Basic { instructions, .. } = block {
                let num_insts = instructions.len();
                a.record_block_fact(id, num_insts, &init);
            }
        })
        .collect();

    if a.is_forward() {
        // Reverse post-order for forward analysis.
        worklist.make_contiguous().reverse();
    }

    while let Some(block) = worklist.pop_front() {
        let id = block.id();
        // Ensures we reflect the state of the `worklist`.
        seen_blocks.remove(&id);

        let old_block_fact = a.get_block_fact(id).cloned();

        let incoming = a.meet(block, &init);
        a.transfer(block, incoming);

        if old_block_fact.as_ref() != a.get_block_fact(id)
            && let Block::Basic {
                successors,
                predecessors,
                ..
            } = block
        {
            let next_blocks = if a.is_forward() {
                successors
            } else {
                predecessors
            };

            for id in next_blocks {
                match *id {
                    id if Block::ENTRY_ID == id => {
                        assert!(
                            !a.is_forward(),
                            "malformed control-flow graph: basic block should not have entry as it's successor"
                        );
                    }
                    id if cfg.exit_block_id() == id => {
                        assert!(
                            a.is_forward(),
                            "malformed control-flow graph: basic block should not have exit as it's predecessor"
                        );
                    }
                    id => {
                        // NOTE: O(n) time complexity.
                        if let Some(next_block) = cfg.blocks.iter().find(|block| block.id() == id)
                            && seen_blocks.insert(next_block.id())
                        {
                            worklist.push_back(next_block);
                        }
                    }
                }
            }
        }
    }
}
