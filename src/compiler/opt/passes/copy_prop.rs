//! Copy Propagation
//!
//! Transforms an intermediate representation (_IR_) by replacing variables with
//! their assigned values where applicable, reducing redundant copies.

use std::collections::{HashMap, HashSet};

use crate::compiler::ir::{Instruction, Value};
use crate::compiler::opt::analysis::{DataFlowAnalysis, run_analysis};
use crate::compiler::opt::{Block, CFG, CFGInstruction};

/// TODO: Use `HashMap` to avoid O(n) lookup.
///
/// Tracks copies from `dst` -> `src` (`dst` receives value from `src`).
type ReachingCopies<'a> = HashSet<(Value<'a>, Value<'a>)>;

/// Reaching copies analysis over a control-flow graph.
#[derive(Debug)]
struct CopyProp<'a> {
    copies: HashMap<usize, (ReachingCopies<'a>, Vec<ReachingCopies<'a>>)>,
    exit_id: usize,
}

impl<'a, I> DataFlowAnalysis<I> for CopyProp<'a>
where
    I: CFGInstruction<Instr = Instruction<'a>>,
{
    type Fact = ReachingCopies<'a>;
    type BlockFact = HashMap<usize, (Self::Fact, Vec<Self::Fact>)>;

    fn transfer(&mut self, block: &Block<I>, mut incoming: Self::Fact) {
        if let Block::Basic {
            id: block_id,
            instructions,
            ..
        } = block
        {
            for (i, instr) in instructions.iter().enumerate() {
                // Record the set of copies that reach the point before the
                // current instruction.
                <CopyProp<'_> as DataFlowAnalysis<I>>::record_instruction_fact(
                    self, *block_id, i, &incoming,
                );

                // Compute the set of copies that reach the point after the
                // current instruction.
                match instr.concrete() {
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
                    Instruction::FnCall { dst, .. } => {
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

            <CopyProp<'_> as DataFlowAnalysis<I>>::record_block_fact(
                self,
                block.id(),
                instructions.len(),
                &incoming,
            );
        }
    }

    fn meet(&self, block: &Block<I>, initial: &Self::Fact) -> Self::Fact {
        let mut incoming = initial.clone();

        if let Block::Basic { predecessors, .. } = block {
            for id in predecessors {
                match *id {
                    // Since no copies reach the start of the function (entry),
                    // return an empty set. The intersection of any set with
                    // the empty set is still the empty set.
                    id if Block::<I>::ENTRY_ID == id => return Self::Fact::default(),
                    id => {
                        assert!(
                            self.exit_id != id,
                            "malformed control-flow graph: basic block should not have exit as it's predecessor"
                        );

                        if let Some(pred_outgoing) =
                            <CopyProp<'_> as DataFlowAnalysis<I>>::get_block_fact(self, id)
                        {
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

    fn initial(cfg: &CFG<I>) -> Self::Fact {
        let mut initial = Self::Fact::default();

        for block in &cfg.basic_blocks() {
            if let Block::Basic { instructions, .. } = block {
                for inst in instructions {
                    if let Instruction::Copy { src, dst } = inst.concrete() {
                        initial.insert((dst.clone(), src.clone()));
                    }
                }
            }
        }

        initial
    }

    #[inline]
    fn block_facts(&self) -> &Self::BlockFact {
        &self.copies
    }

    #[inline]
    fn block_facts_mut(&mut self) -> &mut Self::BlockFact {
        &mut self.copies
    }

    #[inline]
    fn is_forward(&self) -> bool {
        true
    }
}

/// Transforms a control-flow graph (_CFG_) by replacing variables with their
/// assigned values where applicable, reducing redundant copies.
pub fn propagate_copy<'a, 'b, I>(cfg: &'a mut CFG<I>)
where
    I: CFGInstruction<Instr = Instruction<'b>>,
{
    let mut copy_prop = CopyProp {
        exit_id: cfg.exit_block_id(),
        copies: HashMap::default(),
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
                if let Some(copies) = <CopyProp<'_> as DataFlowAnalysis<I>>::get_instruction_fact(
                    &copy_prop, *block_id, i,
                ) {
                    match inst.concrete_mut() {
                        Instruction::Copy { src, dst } => {
                            // Instruction has no affect if `src` and `dst`
                            // already have the same value with copies that
                            // reach this instruction.
                            if copies.contains(&(dst.clone(), src.clone()))
                                || copies.contains(&(src.clone(), dst.clone()))
                            {
                                to_remove.push(i);
                                continue;
                            }

                            rewrite_operand(src, copies);
                        }
                        Instruction::Unary { src, .. } | Instruction::Return(src) => {
                            rewrite_operand(src, copies);
                        }
                        Instruction::Binary { lhs, rhs, .. } => {
                            rewrite_operand(lhs, copies);
                            rewrite_operand(rhs, copies);
                        }
                        Instruction::FnCall { args, .. } => {
                            for arg in args {
                                rewrite_operand(arg, copies);
                            }
                        }
                        Instruction::JumpIfZero { cond, .. }
                        | Instruction::JumpIfNotZero { cond, .. } => {
                            rewrite_operand(cond, copies);
                        }
                        _ => {}
                    }
                }
            }

            // Removing instructions from right-left ensures indicies are not
            // affected by shifting.
            for i in to_remove.drain(..).rev() {
                // NOTE: O(n) time complexity.
                instructions.remove(i);
            }
        }
    }
}

/// Rewrites the given _IR_ value using the reaching copies at the current
/// instruction.
fn rewrite_operand<'a>(val: &mut Value<'a>, copies: &ReachingCopies<'a>) {
    if let Value::Var { .. } = val
        // NOTE: O(n) time complexity.
        && let Some((_, src)) = copies.iter().find(|(dst, _)| dst == val)
    {
        val.clone_from(src);
    }
}
