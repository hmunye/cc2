//! Dead Store Elimination (DSE)
//!
//! Transforms an intermediate representation (_IR_) by removing assignments to
//! variables that are never used or updated.

use std::collections::{HashMap, HashSet};

use crate::compiler::ir::{Instruction, Value};
use crate::compiler::opt::analysis::{DataFlowAnalysis, run_analysis};
use crate::compiler::opt::{Block, CFG, CFGInstruction};

/// Tracks the set of live variables at a program point using internal IDs.
type Stores = HashSet<usize>;

/// Liveness analysis over a control-flow graph.
#[derive(Debug)]
struct DeadStore {
    /// Mapping from block ID to live variables at the block's exit and the
    /// per-instruction live variables by index.
    stores: HashMap<usize, (Stores, Vec<Stores>)>,
    statics: Stores,
    exit_id: usize,
}

impl<'a, I> DataFlowAnalysis<I> for DeadStore
where
    I: CFGInstruction<Instr = Instruction<'a>>,
{
    type Fact = Stores;
    type BlockFact = HashMap<usize, (Self::Fact, Vec<Self::Fact>)>;

    fn transfer(&mut self, block: &Block<I>, mut outgoing: Self::Fact) {
        if let Block::Basic {
            id: block_id,
            instructions,
            ..
        } = block
        {
            for (i, instr) in instructions.iter().enumerate().rev() {
                // Record the set of live variables that reach the point after
                // the current instruction.
                <DeadStore as DataFlowAnalysis<I>>::record_instruction_fact(
                    self, *block_id, i, &outgoing,
                );

                // Compute the set of live variables that reach the point before
                // the current instruction.
                match instr.concrete() {
                    Instruction::Binary { lhs, rhs, dst, .. } => {
                        if let Value::Var { id, .. } = dst {
                            // Remove `dst` from the live set because this
                            // instruction defines it, killing the previously
                            // live value.
                            outgoing.remove(id);
                        }

                        if let Value::Var { id, .. } = lhs {
                            outgoing.insert(*id);
                        }

                        if let Value::Var { id, .. } = rhs {
                            outgoing.insert(*id);
                        }
                    }
                    Instruction::Unary { src, dst, .. } | Instruction::Copy { src, dst } => {
                        if let Value::Var { id, .. } = dst {
                            // Remove `dst` from the live set because this
                            // instruction defines it, killing the previously
                            // live value.
                            outgoing.remove(id);
                        }

                        if let Value::Var { id, .. } = src {
                            outgoing.insert(*id);
                        }
                    }
                    Instruction::JumpIfZero { cond: val, .. }
                    | Instruction::JumpIfNotZero { cond: val, .. }
                    | Instruction::Return(val) => {
                        if let Value::Var { id, .. } = val {
                            outgoing.insert(*id);
                        }
                    }
                    Instruction::FnCall { args, dst, .. } => {
                        if let Value::Var { id, .. } = dst {
                            // Remove `dst` from the live set because this
                            // instruction defines it, killing the previously
                            // live value.
                            outgoing.remove(id);
                        }

                        args.iter().filter_map(Value::as_id).for_each(|id| {
                            outgoing.insert(id);
                        });

                        // Interprocedural analysis is not performed, so also
                        // conservatively mark all static variables as live,
                        // since they may be read and/or updated across function
                        // boundaries.
                        outgoing.extend(self.statics.iter().copied());
                    }
                    _ => {}
                }
            }

            <DeadStore as DataFlowAnalysis<I>>::record_block_fact(
                self,
                block.id(),
                instructions.len(),
                &outgoing,
            );
        }
    }

    fn meet(&self, block: &Block<I>, initial: &Self::Fact) -> Self::Fact {
        let mut outgoing = initial.clone();

        if let Block::Basic { successors, .. } = block {
            for id in successors {
                match *id {
                    // All static variables used within the function are live
                    // at the exit block (end of function).
                    id if self.exit_id == id => {
                        outgoing.extend(self.statics.iter().copied());
                    }
                    id => {
                        assert!(
                            Block::<I>::ENTRY_ID != id,
                            "malformed control-flow graph: basic block should not have entry as it's successor"
                        );

                        if let Some(succ_incoming) =
                            <DeadStore as DataFlowAnalysis<I>>::get_block_fact(self, id)
                        {
                            outgoing.extend(succ_incoming.iter().copied());
                        }
                    }
                }
            }
        }

        outgoing
    }

    #[inline]
    fn initial(&mut self, _cfg: &CFG<I>) -> Self::Fact {
        // The identity element for the `meet` operator (union) is the empty set
        // (at the function exit, no local variables are live).
        Self::Fact::default()
    }

    #[inline]
    fn block_facts(&self) -> &Self::BlockFact {
        &self.stores
    }

    #[inline]
    fn block_facts_mut(&mut self) -> &mut Self::BlockFact {
        &mut self.stores
    }

    #[inline]
    fn is_forward(&self) -> bool {
        false
    }
}

/// Transforms a control-flow graph by removing assignments to variables that
/// are never used or updated.
pub fn dead_store<'a, 'b, I>(cfg: &mut CFG<I>)
where
    I: CFGInstruction<Instr = Instruction<'b>>,
{
    let mut dse = DeadStore {
        stores: HashMap::default(),
        statics: collect_statics(cfg),
        exit_id: cfg.exit_block_id(),
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
                if let Some(stores) =
                    <DeadStore as DataFlowAnalysis<I>>::get_instruction_fact(&dse, *block_id, i)
                {
                    match inst.concrete() {
                        Instruction::Copy { dst, .. }
                        | Instruction::Unary { dst, .. }
                        | Instruction::Binary { dst, .. } => {
                            if let Some(id) = dst.as_id()
                                && !stores.contains(&id)
                            {
                                to_remove.push(i);
                            }
                        }
                        // Since interprocedural analysis is not performed, function
                        // calls are ignored, as they may have side-effects.
                        _ => {}
                    }
                }
            }

            // Removing instructions from right-left ensures indicies are not
            // affected by shifting.
            for i in to_remove.drain(..).rev() {
                instructions.remove(i);
            }
        }
    }
}

fn collect_statics<'a, I>(cfg: &CFG<I>) -> Stores
where
    I: CFGInstruction<Instr = Instruction<'a>>,
{
    let mut statics = HashSet::default();

    for block in &cfg.basic_blocks() {
        if let Block::Basic { instructions, .. } = block {
            for inst in instructions {
                match inst.concrete() {
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
                    Instruction::FnCall { args, dst, .. } => {
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
fn insert_static(val: &Value<'_>, statics: &mut Stores) {
    if val.is_static()
        && let Some(id) = val.as_id()
    {
        statics.insert(id);
    }
}
