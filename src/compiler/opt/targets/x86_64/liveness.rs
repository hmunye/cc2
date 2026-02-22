use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use crate::compiler::mir::{Instruction, Reg};
use crate::compiler::opt::analysis::DataFlowAnalysis;
use crate::compiler::opt::targets::x86_64::RegisterType;
use crate::compiler::opt::{Block, CFG, CFGInstruction};
use crate::compiler::parser::sema::symbols::SymbolMap;

/// Tracks the set of live registers at a program point.
type LiveRegs<'a> = HashSet<RegisterType<'a>>;

/// Liveness analysis for register allocation over a control-flow graph.
#[derive(Debug)]
pub struct RegisterLiveness<'a> {
    /// Mapping from block ID to live registers at the block's exit and the
    /// per-instruction live registers by index.
    pub lives: HashMap<usize, (LiveRegs<'a>, Vec<LiveRegs<'a>>)>,
    sym_map: &'a SymbolMap,
    exit_id: usize,
}

impl<'a> RegisterLiveness<'a> {
    /// Returns a new `RegisterLiveness`.
    #[inline]
    #[must_use]
    pub fn new(exit_id: usize, sym_map: &'a SymbolMap) -> Self {
        Self {
            lives: HashMap::default(),
            sym_map,
            exit_id,
        }
    }

    // TODO: Could possibly be a default implementation using an associated type.
    //
    /// Returns the set of live registers just after instruction `inst_id` in
    /// block `block_id`, or `None` if the block containing the instruction has
    /// not been initialized.
    #[inline]
    #[must_use]
    pub fn get_instruction_fact(&self, block_id: usize, inst_id: usize) -> Option<&LiveRegs<'_>> {
        let entry = self.lives.get(&block_id)?;
        Some(&entry.1[inst_id])
    }

    // TODO: Could possibly be a default implementation using an associated type.
    //
    /// Records the set of live registers just after the instruction `inst_id`
    /// in the block `block_id`.
    #[inline]
    fn record_instruction_fact(&mut self, block_id: usize, inst_id: usize, lives: &LiveRegs<'a>) {
        let entry = self
            .lives
            .get_mut(&block_id)
            .expect("block of instruction must be initialized before recording facts");

        let slot = &mut entry.1[inst_id];

        // Reuses existing allocation when possible.
        slot.clone_from(lives);
    }
}

impl<'a, 'b, I> DataFlowAnalysis<'a, I> for RegisterLiveness<'b>
where
    I: CFGInstruction<Instr = Instruction<'b>>,
{
    type Fact = LiveRegs<'b>;

    fn transfer(&mut self, block: &Block<I>, mut outgoing: Self::Fact) {
        if let Block::Basic {
            id: block_id,
            instructions,
            ..
        } = block
        {
            let mut used = vec![];
            let mut updated = vec![];

            // Compute the set of live registers that reach the point before
            // the current instruction.
            for (i, instr) in instructions.iter().enumerate().rev() {
                // Record the set of live registers that reach the point after
                // the current instruction.
                self.record_instruction_fact(*block_id, i, &outgoing);

                instr
                    .concrete()
                    .find_used_and_updated(&mut used, &mut updated, self.sym_map);

                updated
                    .drain(..)
                    .filter_map(|op| op.try_into().ok())
                    .for_each(|reg| {
                        outgoing.remove(&reg);
                    });

                used.drain(..)
                    .filter_map(|op| op.try_into().ok())
                    .for_each(|reg| {
                        outgoing.insert(reg);
                    });
            }

            <RegisterLiveness<'_> as DataFlowAnalysis<'_, I>>::record_block_fact(
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
                    // Ensure the `%AX` register is marked live at the block
                    // exit, since it is used to store the return value.
                    id if self.exit_id == id => {
                        outgoing.insert(RegisterType::Hardware(Reg::AX));
                    }
                    id => {
                        assert!(
                            Block::<I>::ENTRY_ID != id,
                            "malformed control-flow graph: basic block should not have entry as it's successor"
                        );

                        if let Some(succ_incoming) =
                            <RegisterLiveness<'_> as DataFlowAnalysis<'_, I>>::get_block_fact(
                                self, id,
                            )
                        {
                            outgoing.extend(succ_incoming.iter().copied());
                        }
                    }
                }
            }
        }

        outgoing
    }

    fn initial(&self, _cfg: &'a CFG<I>) -> Self::Fact {
        // The identity element for the `meet` operator (union) is the empty set
        // (at stack-frame exit, no registers are live, ignoring callee-saved
        // registers which are handled separately).
        Self::Fact::default()
    }

    // TODO: Could possibly be a default implementation using an associated type.
    #[inline]
    fn record_block_fact(&mut self, block_id: usize, num_insts: usize, fact: &Self::Fact) {
        match self.lives.entry(block_id) {
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

    // TODO: Could possibly be a default implementation using an associated type.
    #[inline]
    fn get_block_fact(&self, block_id: usize) -> Option<&Self::Fact> {
        let entry = self.lives.get(&block_id)?;
        Some(&entry.0)
    }

    #[inline]
    fn is_forward(&self) -> bool {
        false
    }
}
