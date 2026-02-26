//! Unreachable Code Elimination (UCE)
//!
//! Transforms an intermediate representation (_IR_) by removing code that can
//! never be executed based on control flow analysis.

use std::collections::HashSet;

use crate::compiler::ir::Instruction;
use crate::compiler::opt::{Block, CFG, CFGInstruction};

/// Transforms the provided control-flow graph, removing code that can never be
/// executed.
pub fn unreachable_code<'a, I>(cfg: &mut CFG<I>)
where
    I: CFGInstruction<Instr = Instruction<'a>>,
{
    let mut reachable: HashSet<_> = cfg.basic_blocks().post_order().map(Block::id).collect();

    // Ensure entry and exit blocks are not removed.
    reachable.insert(Block::<I>::ENTRY_ID);
    reachable.insert(cfg.exit_block_id());

    // Remove all unreachable basic blocks from the graph while preserving their
    // relative order.
    cfg.blocks.retain(|block| reachable.contains(&block.id()));

    clean_cfg(cfg, &reachable);
}

/// Removes redundant jump instructions, useless label instructions, and
/// unreachable predecessors from the control-flow graph.
///
/// Redundant jumps are those that only jump to the next block in sequence.
/// Useless labels are those that are not targeted by any block but the
/// previous.
fn clean_cfg<'a, I>(cfg: &mut CFG<I>, reachable: &HashSet<usize>)
where
    I: CFGInstruction<Instr = Instruction<'a>>,
{
    let exit_id = cfg.exit_block_id();

    // Iterate over the blocks (excluding entry block).
    for i in Block::<I>::ENTRY_ID + 1..cfg.blocks.len() {
        // Keep mapping from block ID to index up to date.
        cfg.id_to_index
            .entry(cfg.blocks[i].id())
            .and_modify(|index| *index = i);

        let prev_block_id = cfg.blocks[i - 1].id();

        let next_block_id = if let Some(block) = cfg.blocks.get(i + 1) {
            block.id()
        } else {
            // At the exit block: continue;
            continue;
        };

        let block = &mut cfg.blocks[i];

        if let Block::Basic {
            instructions,
            predecessors,
            successors,
            ..
        } = block
        {
            // Retain only the reachable predecessors.
            predecessors.retain(|id| reachable.contains(id));

            if let Some(
                Instruction::Jump(_)
                | Instruction::JumpIfZero { .. }
                | Instruction::JumpIfNotZero { .. },
            ) = instructions.last().map(CFGInstruction::concrete)
            {
                // Skip the last basic block since a `jump` instruction at
                // the end of the graph targets the exit block.
                if next_block_id == exit_id {
                    continue;
                }

                let mut keep = false;

                for id in successors {
                    // There is a successor other than the next block where
                    // control can flow to.
                    if *id != next_block_id {
                        keep = true;
                        break;
                    }
                }

                if !keep {
                    instructions.pop();
                }
            }

            if let Some(Instruction::Label(_)) = instructions.first().map(CFGInstruction::concrete)
            {
                let mut keep = false;

                for id in predecessors {
                    // There is a predecessor other than the previous block
                    // which targets this label.
                    if *id != prev_block_id {
                        keep = true;
                        break;
                    }
                }

                if !keep {
                    // NOTE: O(n) time complexity.
                    instructions.remove(0);
                }
            }
        }
    }

    if let Some(Block::Exit { predecessors, .. }) = cfg.blocks.last_mut() {
        // Retain only the reachable predecessors.
        predecessors.retain(|id| reachable.contains(id));
    }
}
