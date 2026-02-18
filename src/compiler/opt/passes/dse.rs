//! Dead Store Elimination (DSE)
//!
//! Transforms an intermediate representation (_IR_) by removing assignments to
//! variables that are never used or updated.

// TODO: Update IR to include unique ID for each variable, so strings do not
// have to be cloned.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use crate::compiler::ir::{Instruction, Value};
use crate::compiler::opt::analysis::{DataFlowAnalysis, run_analysis};
use crate::compiler::opt::{Block, CFG};

/// Tracks the set of live variables at a program point.
type Stores = HashSet<String>;

/// Liveness analysis over a control-flow graph.
#[derive(Debug)]
struct DeadStore {
    exit_id: usize,
    statics: Stores,
    /// Mapping from block ID to live variables at the block's exit and the
    /// per-instruction live variables by index.
    stores: HashMap<usize, (Stores, Vec<Stores>)>,
}

impl DeadStore {
    /// Records the set of live variables just after the instruction `inst_id`
    /// in the block `block_id`.
    #[inline]
    fn record_instruction_fact(&mut self, block_id: usize, inst_id: usize, stores: &Stores) {
        let entry = self
            .stores
            .get_mut(&block_id)
            .expect("block of instruction must be initialized before recording facts");

        let slot = &mut entry.1[inst_id];

        // Reuses existing allocation when possible.
        slot.clone_from(stores);
    }

    /// Returns the set of live variables just after instruction `inst_id` in
    /// block `block_id`.
    #[inline]
    fn get_instruction_fact(&self, block_id: usize, inst_id: usize) -> &Stores {
        let entry = self
            .stores
            .get(&block_id)
            .expect("block of instruction must be initialized before getting facts");

        &entry.1[inst_id]
    }
}

impl<'a> DataFlowAnalysis<'a> for DeadStore {
    type Fact = Stores;

    fn transfer(&mut self, block: &Block<'a>, mut outgoing: Self::Fact) {
        if let Block::Basic {
            id: block_id,
            instructions,
            ..
        } = block
        {
            for (i, inst) in instructions.iter().enumerate().rev() {
                // Record the set of live variables that reach the point after
                // the current instruction.
                self.record_instruction_fact(*block_id, i, &outgoing);

                // Compute the set of live variables that reach the point before
                // the current instruction.
                match &inst {
                    Instruction::Binary { lhs, rhs, dst, .. } => {
                        if let Some(ident) = dst.as_var() {
                            // Remove `dst` from the live set because this
                            // instruction defines it, killing the previously
                            // live value.
                            outgoing.remove(ident);
                        }

                        if let Some(ident) = lhs.as_var() {
                            outgoing.insert(ident.to_string());
                        }

                        if let Some(ident) = rhs.as_var() {
                            outgoing.insert(ident.to_string());
                        }
                    }
                    Instruction::Unary { src, dst, .. } | Instruction::Copy { src, dst } => {
                        if let Some(ident) = dst.as_var() {
                            // Remove `dst` from the live set because this
                            // instruction defines it, killing the previously
                            // live value.
                            outgoing.remove(ident);
                        }

                        if let Some(ident) = src.as_var() {
                            outgoing.insert(ident.to_string());
                        }
                    }
                    Instruction::JumpIfZero { cond: val, .. }
                    | Instruction::JumpIfNotZero { cond: val, .. }
                    | Instruction::Return(val) => {
                        if let Some(ident) = val.as_var() {
                            outgoing.insert(ident.to_string());
                        }
                    }
                    Instruction::Call { args, dst, .. } => {
                        if let Some(ident) = dst.as_var() {
                            // Remove `dst` from the live set because this
                            // instruction defines it, killing the previously
                            // live value.
                            outgoing.remove(ident);
                        }

                        args.iter().filter_map(Value::as_var).for_each(|ident| {
                            outgoing.insert(ident.to_string());
                        });

                        // Interprocedural analysis is not performed, so also
                        // conservatively mark all static variables as live,
                        // since they may be read and/or updated across function
                        // boundaries.
                        outgoing.extend(self.statics.iter().cloned());
                    }
                    _ => {}
                }
            }

            self.record_block_fact(block.id(), instructions.len(), &outgoing);
        }
    }

    fn meet(&self, block: &Block<'_>, initial: &Self::Fact) -> Self::Fact {
        let mut outgoing = initial.clone();

        if let Block::Basic { successors, .. } = block {
            for id in successors {
                match *id {
                    // All static variables used within the function are live
                    // at the exit block (end of function).
                    id if self.exit_id == id => {
                        outgoing.extend(self.statics.iter().cloned());
                    }
                    id => {
                        assert!(
                            Block::ENTRY_ID != id,
                            "malformed control-flow graph: basic block should not have entry as it's successor"
                        );

                        if let Some(succ_incoming) = self.get_block_fact(id) {
                            outgoing.extend(succ_incoming.iter().cloned());
                        }
                    }
                }
            }
        }

        outgoing
    }

    fn initial(&self, _cfg: &'a CFG<'a>) -> Self::Fact {
        // The identity element for the `meet` operator (union) is the empty set
        // (at the function exit, no local variables are live).
        Self::Fact::default()
    }

    #[inline]
    fn record_block_fact(&mut self, block_id: usize, num_insts: usize, fact: &Self::Fact) {
        match self.stores.entry(block_id) {
            Entry::Occupied(mut entry) => {
                let exit_fact = &mut entry.get_mut().0;
                // Reuses existing allocation when possible.
                exit_fact.clone_from(fact);
            }
            Entry::Vacant(entry) => {
                entry.insert((fact.clone(), vec![Self::Fact::default(); num_insts]));
            }
        }
    }

    #[inline]
    fn get_block_fact(&self, block_id: usize) -> Option<&Self::Fact> {
        let entry = self.stores.get(&block_id)?;
        Some(&entry.0)
    }

    #[inline]
    fn is_forward(&self) -> bool {
        false
    }
}

/// Transforms a control-flow graph by removing assignments to variables that
/// are never used or updated.
pub fn dead_store(cfg: &mut CFG<'_>) {
    let mut dse = DeadStore {
        exit_id: cfg.exit_block_id(),
        statics: collect_statics(cfg),
        stores: HashMap::default(),
    };

    run_analysis(cfg, &mut dse);

    let mut to_remove = Vec::new();

    for block in cfg.basic_blocks_mut() {
        if let Block::Basic {
            id: block_id,
            instructions,
            ..
        } = block
        {
            for (i, inst) in instructions.iter().enumerate() {
                let stores = dse.get_instruction_fact(*block_id, i);

                // Since interprocedural analysis is not performed, function
                // calls are ignored, as they may have side-effects.
                match inst {
                    Instruction::Copy { dst, .. }
                    | Instruction::Unary { dst, .. }
                    | Instruction::Binary { dst, .. } => {
                        if let Some(ident) = dst.as_var()
                            && !stores.contains(ident)
                        {
                            to_remove.push(i);
                        }
                    }
                    _ => {}
                }
            }

            // Removing instructions from right-left ensures indicies are not
            // affected by shifting.
            #[allow(clippy::iter_with_drain)]
            for i in to_remove.drain(..).rev() {
                // NOTE: O(n) time complexity.
                instructions.remove(i);
            }
        }
    }
}

fn collect_statics(cfg: &CFG<'_>) -> Stores {
    let mut statics = HashSet::default();

    for block in &cfg.basic_blocks() {
        if let Block::Basic { instructions, .. } = block {
            for inst in instructions {
                match inst {
                    Instruction::Binary { lhs, rhs, dst, .. } => {
                        insert_static(dst, &mut statics);
                        insert_static(lhs, &mut statics);
                        insert_static(rhs, &mut statics);
                    }
                    Instruction::Unary { src, dst, .. } | Instruction::Copy { src, dst } => {
                        insert_static(dst, &mut statics);
                        insert_static(src, &mut statics);
                    }
                    Instruction::JumpIfZero { cond: val, .. }
                    | Instruction::JumpIfNotZero { cond: val, .. }
                    | Instruction::Return(val) => {
                        insert_static(val, &mut statics);
                    }
                    Instruction::Call { args, dst, .. } => {
                        insert_static(dst, &mut statics);

                        for arg in args {
                            insert_static(arg, &mut statics);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    statics
}

#[inline]
fn insert_static(val: &Value, statics: &mut Stores) {
    if val.is_static()
        && let Some(ident) = val.as_var()
    {
        statics.insert(ident.to_string());
    }
}
