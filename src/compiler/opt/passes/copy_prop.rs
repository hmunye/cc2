//! Copy Propagation
//!
//! Transforms an intermediate representation (_IR_) by replacing variables with
//! their assigned values where applicable, reducing redundant copies.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use crate::compiler::ir::{Instruction, Value};
use crate::compiler::opt::analysis::{DataFlowAnalysis, run_analysis};
use crate::compiler::opt::{Block, CFG};

/// Tracks copies from `dst` -> `src` (`dst` receives value from `src`).
type ReachingCopies = HashSet<(Value, Value)>;

/// Reaching copies analysis over a control-flow graph.
#[derive(Debug)]
struct CopyProp {
    exit_id: usize,
    /// Mapping from block ID to reaching copies at the block's exit and the
    /// per-instruction reaching copies by index.
    reaching_copies: HashMap<usize, (ReachingCopies, Vec<ReachingCopies>)>,
}

impl CopyProp {
    /// Records the set of copies reaching just before the instruction `inst_id`
    /// in the block `block_id`.
    #[inline]
    fn record_instruction_fact(
        &mut self,
        block_id: usize,
        inst_id: usize,
        copies: &ReachingCopies,
    ) {
        let entry = self
            .reaching_copies
            .get_mut(&block_id)
            .expect("block of instruction must be initialized before recording facts");

        let slot = &mut entry.1[inst_id];

        // Reuses existing allocation when possible.
        slot.clone_from(copies);
    }

    /// Returns the set of copies reaching just before instruction `inst_id` in
    /// block `block_id`, or `None` if the block containing the instruction has
    /// not been initialized.
    #[inline]
    fn get_instruction_fact(&self, block_id: usize, inst_id: usize) -> Option<&ReachingCopies> {
        let entry = self.reaching_copies.get(&block_id)?;
        Some(&entry.1[inst_id])
    }
}

impl<'a> DataFlowAnalysis<'a> for CopyProp {
    type Fact = ReachingCopies;

    fn transfer(&mut self, block: &Block<'a>, mut incoming: Self::Fact) {
        if let Block::Basic {
            id: block_id,
            instructions,
            ..
        } = block
        {
            for (i, inst) in instructions.iter().enumerate() {
                // Record the set of copies that reach the point before the
                // current instruction.
                self.record_instruction_fact(*block_id, i, &incoming);

                // Compute the set of copies that reach the point after the
                // current instruction.
                match &inst {
                    Instruction::Copy { src, dst } => {
                        if incoming.contains(&(src.clone(), dst.clone())) {
                            // Skip trivial copy (e.g. `x = y` copy after prior
                            // `y = x` copy was recorded).
                            continue;
                        }

                        // Kill any copies to and from `dst` before recording
                        // it.
                        incoming.retain(|copy| copy.0 != *dst && copy.1 != *dst);

                        incoming.insert((dst.clone(), src.clone()));
                    }
                    Instruction::Call { dst, .. } => {
                        // Kill any copies to and from `dst`.
                        //
                        // Interprocedural analysis is not performed, so also
                        // conservatively remove copies using static variables
                        // that may be used across function boundaries.
                        incoming.retain(|copy| {
                            (!copy.0.is_static() && copy.0 != *dst)
                                && (!copy.1.is_static() && copy.1 != *dst)
                        });
                    }
                    Instruction::Unary { dst, .. } | Instruction::Binary { dst, .. } => {
                        // Kill any copies to and from `dst`.
                        incoming.retain(|copy| copy.0 != *dst && copy.1 != *dst);
                    }
                    _ => {}
                }
            }

            self.record_block_fact(block.id(), instructions.len(), &incoming);
        }
    }

    fn meet(&self, block: &Block<'_>, initial: &Self::Fact) -> Self::Fact {
        let mut incoming = initial.clone();

        if let Block::Basic { predecessors, .. } = block {
            for id in predecessors {
                match *id {
                    // Since no copies reach the start of the function (entry),
                    // return an empty set. The intersection of any set with
                    // the empty set is still the empty set.
                    id if Block::ENTRY_ID == id => return Self::Fact::default(),
                    id => {
                        assert!(
                            self.exit_id != id,
                            "malformed control-flow graph: basic block should not have exit as it's predecessor"
                        );

                        if let Some(pred_outgoing) = self.get_block_fact(id) {
                            // Retain those copies that intersect with the
                            // predecessors copies.
                            incoming.retain(|copy| pred_outgoing.contains(copy));
                        }
                    }
                }
            }
        }

        incoming
    }

    fn initial(&self, cfg: &'a CFG<'a>) -> Self::Fact {
        let mut initial = Self::Fact::default();

        for block in &cfg.basic_blocks() {
            if let Block::Basic { instructions, .. } = block {
                for inst in instructions {
                    if let Instruction::Copy { src, dst } = inst {
                        initial.insert((dst.clone(), src.clone()));
                    }
                }
            }
        }

        initial
    }

    #[inline]
    fn record_block_fact(&mut self, block_id: usize, num_insts: usize, fact: &Self::Fact) {
        match self.reaching_copies.entry(block_id) {
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
        let entry = self.reaching_copies.get(&block_id)?;
        Some(&entry.0)
    }

    #[inline]
    fn is_forward(&self) -> bool {
        true
    }
}

/// Transforms a control-flow graph (_CFG_) by replacing variables with their
/// assigned values where applicable, reducing redundant copies.
pub fn propagate_copy(cfg: &mut CFG<'_>) {
    let mut copy_prop = CopyProp {
        exit_id: cfg.exit_block_id(),
        reaching_copies: HashMap::default(),
    };

    run_analysis(cfg, &mut copy_prop);

    let mut to_remove = Vec::new();

    for block in cfg.basic_blocks_mut() {
        if let Block::Basic {
            id: block_id,
            instructions,
            ..
        } = block
        {
            for (i, inst) in instructions.iter_mut().enumerate() {
                if let Some(reaching_copies) = copy_prop.get_instruction_fact(*block_id, i) {
                    match inst {
                        Instruction::Copy { src, dst } => {
                            // Instruction has no affect if `src` and `dst`
                            // already have the same value with copies that
                            // reach this instruction.
                            if reaching_copies.contains(&(dst.clone(), src.clone()))
                                || reaching_copies.contains(&(src.clone(), dst.clone()))
                            {
                                to_remove.push(i);
                                continue;
                            }

                            rewrite_operand(src, reaching_copies);
                        }
                        Instruction::Unary { src, .. } | Instruction::Return(src) => {
                            rewrite_operand(src, reaching_copies);
                        }
                        Instruction::Binary { lhs, rhs, .. } => {
                            rewrite_operand(lhs, reaching_copies);
                            rewrite_operand(rhs, reaching_copies);
                        }
                        Instruction::Call { args, .. } => {
                            for arg in args {
                                rewrite_operand(arg, reaching_copies);
                            }
                        }
                        Instruction::JumpIfZero { cond, .. }
                        | Instruction::JumpIfNotZero { cond, .. } => {
                            rewrite_operand(cond, reaching_copies);
                        }
                        _ => {}
                    }
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

/// Rewrites the given _IR_ value using the reaching copies at the current
/// instruction.
fn rewrite_operand(val: &mut Value, reaching_copies: &ReachingCopies) {
    if let Value::Var { .. } = val
        // NOTE: O(n) time complexity.
        && let Some((_, src)) = reaching_copies.iter().find(|(dst, _)| dst == val)
    {
        val.clone_from(src);
    }
}
