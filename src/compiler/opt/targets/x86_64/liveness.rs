use std::collections::{HashMap, HashSet};

use crate::compiler::frontend::SymbolTable;
use crate::compiler::opt::analysis::DataFlowAnalysis;
use crate::compiler::opt::targets::x86_64::RegisterType;
use crate::compiler::opt::{Block, CFG, CFGInstruction};
use crate::compiler::targets::x86_64::{Instruction, Reg};

/// Tracks the set of live registers at a program point.
type LiveRegs<'a> = HashSet<RegisterType<'a>>;

/// Liveness analysis for register allocation over a control-flow graph.
#[derive(Debug)]
pub struct RegisterLiveness<'a> {
    /// Mapping from block ID to live registers at the block's exit and the
    /// per-instruction live registers by index.
    pub lives: HashMap<usize, (LiveRegs<'a>, Vec<LiveRegs<'a>>)>,
    sym_table: &'a SymbolTable,
    exit_id: usize,
}

impl<'a> RegisterLiveness<'a> {
    /// Returns a new `RegisterLiveness`.
    #[inline]
    #[must_use]
    pub fn new(exit_id: usize, sym_table: &'a SymbolTable) -> Self {
        Self {
            lives: HashMap::default(),
            sym_table,
            exit_id,
        }
    }
}

impl<'a, I> DataFlowAnalysis<I> for RegisterLiveness<'a>
where
    I: CFGInstruction<Instr = Instruction<'a>>,
{
    type Fact = LiveRegs<'a>;
    type BlockFact = HashMap<usize, (Self::Fact, Vec<Self::Fact>)>;

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
                <RegisterLiveness<'_> as DataFlowAnalysis<I>>::record_instruction_fact(
                    self, *block_id, i, &outgoing,
                );

                instr
                    .concrete()
                    .find_used_and_updated(&mut used, &mut updated, self.sym_table);

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

            <RegisterLiveness<'_> as DataFlowAnalysis<I>>::record_block_fact(
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
                            <RegisterLiveness<'_> as DataFlowAnalysis<I>>::get_block_fact(self, id)
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
    fn initial(_cfg: &CFG<I>) -> Self::Fact {
        // The identity element for the `meet` operator (union) is the empty set
        // (at stack-frame exit, no registers are live, ignoring callee-saved
        // registers which are handled separately).
        Self::Fact::default()
    }

    #[inline]
    fn block_facts(&self) -> &Self::BlockFact {
        &self.lives
    }

    #[inline]
    fn block_facts_mut(&mut self) -> &mut Self::BlockFact {
        &mut self.lives
    }

    #[inline]
    fn is_forward(&self) -> bool {
        false
    }
}
