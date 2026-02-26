//! Data Flow Analysis
//!
//! Generic framework for bi-directional data-flow analyses over a control-flow
//! graph (_CFG_).

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::compiler::opt::{Block, CFG, CFGInstruction};

/// NOTE: Stop using when associated type defaults are stable:
///
/// <https://github.com/rust-lang/rust/issues/29661>
pub trait MapLike<K, V> {
    fn entry(&mut self, key: K) -> Entry<'_, K, V>;
    fn get(&self, key: &K) -> Option<&V>;
    fn get_mut(&mut self, key: &K) -> Option<&mut V>;
}

impl<K: std::hash::Hash + Eq, V, S: std::hash::BuildHasher> MapLike<K, V> for HashMap<K, V, S> {
    #[inline]
    fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        self.entry(key)
    }

    #[inline]
    fn get(&self, key: &K) -> Option<&V> {
        self.get(key)
    }

    #[inline]
    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.get_mut(key)
    }
}

/// Trait for performing data-flow analysis over a control-flow graph
/// (bi-directional).
pub trait DataFlowAnalysis<I> {
    /// Information being tracked during analysis.
    type Fact: Default + Clone + PartialEq;

    /// Maps block ID to the exit fact and per-instruction facts.
    type BlockFact: MapLike<usize, (Self::Fact, Vec<Self::Fact>)>;

    /// Propagates the incoming/outgoing fact through the block's instructions,
    /// updating the fact for each instruction and the block's exit.
    fn transfer(&mut self, block: &Block<I>, fact: Self::Fact);

    /// Merges facts from multiple execution paths (predecessors or successors),
    /// starting from the given block, returning the intersection of the initial
    /// fact and the facts from neighboring blocks.
    fn meet(&self, block: &Block<I>, initial: &Self::Fact) -> Self::Fact;

    /// Returns the initial facts for analysis (e.g., identity element for
    /// `meet` function).
    fn initial(cfg: &CFG<I>) -> Self::Fact;

    /// Returns a mapping of block IDs to exit and per-instruction facts.
    fn block_facts(&self) -> &Self::BlockFact;

    /// Returns a mutable mapping of block IDs to exit and per-instruction
    /// facts.
    fn block_facts_mut(&mut self) -> &mut Self::BlockFact;

    /// Returns `true` if this is a forward analysis.
    fn is_forward(&self) -> bool;

    /// Stores the fact for the block identified by `id`, replacing any prior
    /// fact.
    fn record_block_fact(&mut self, block_id: usize, num_inst: usize, fact: &Self::Fact) {
        match self.block_facts_mut().entry(block_id) {
            Entry::Occupied(mut entry) => {
                let exit_fact = &mut entry.get_mut().0;
                // Reuses existing allocation when possible.
                exit_fact.clone_from(fact);
            }
            Entry::Vacant(entry) => {
                entry.insert((fact.clone(), vec![Self::Fact::default(); num_inst]));
            }
        }
    }

    /// Returns the recorded fact for the block identified by `id`, or `None` if
    /// no fact has been stored.
    fn get_block_fact<'b>(&'b self, block_id: usize) -> Option<&'b Self::Fact>
    where
        I: 'b,
    {
        let entry = self.block_facts().get(&block_id)?;
        Some(&entry.0)
    }

    /// Stores the fact for the instruction in the given block.
    ///
    /// # Panics
    ///
    /// Panics if the block containing the instruction was not initialized.
    #[inline]
    fn record_instruction_fact(&mut self, block_id: usize, inst_id: usize, fact: &Self::Fact) {
        let entry = self
            .block_facts_mut()
            .get_mut(&block_id)
            .expect("block must have been initialized before recording instruction facts");

        let slot = &mut entry.1[inst_id];

        // Reuses existing allocation when possible.
        slot.clone_from(fact);
    }

    /// Returns the fact for the instruction in the given block, or `None` if
    /// the block is uninitialized.
    #[inline]
    fn get_instruction_fact<'b>(&'b self, block_id: usize, inst_id: usize) -> Option<&'b Self::Fact>
    where
        I: 'b,
    {
        let entry = self.block_facts().get(&block_id)?;
        Some(&entry.1[inst_id])
    }
}

/// Fixed-point solver for a data-flow analysis over a control-flow graph, which
/// iterates over each block until facts converge.
///
/// # Panics
///
/// Panics if the control-flow graph is malformed.
pub fn run_analysis<I: CFGInstruction, A: DataFlowAnalysis<I>>(cfg: &CFG<I>, a: &mut A) {
    let init = A::initial(cfg);

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
                    id if Block::<I>::ENTRY_ID == id => {
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
