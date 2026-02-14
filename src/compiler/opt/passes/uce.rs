//! Unreachable Code Elimination (UCE)
//!
//! Transforms an intermediate representation (_IR_) by removing code that can
//! never be executed based on control flow analysis.

use std::collections::HashSet;

use crate::compiler::ir::Instruction;
use crate::compiler::opt::{Block, CFG};

/// Transforms a control flow graph (_CFG_) by removing code that can never be
/// executed.
pub fn unreachable_code(cfg: &mut CFG<'_>) {
    let entry_id = 0;
    let exit_id = cfg.blocks.len() - 1;

    let mut seen = HashSet::new();

    seen.insert(entry_id);
    seen.insert(exit_id);

    if let Some(Block::Entry { successor, .. }) = &cfg.blocks.first() {
        // NOTE: Make DFS iterative if any issues occur.
        //
        // Start traversal from the successor of the `entry` block.
        mark_reachable_blocks(cfg, *successor, &mut seen);
    }

    // Remove all unreachable basic blocks from the `CFG`.
    cfg.blocks.retain(|block| seen.contains(&block.id()));

    clean_cfg(cfg, &seen);
}

/// Traverses the successors of a basic block given it's `id`, marking all
/// reachable blocks.
fn mark_reachable_blocks(cfg: &CFG<'_>, id: usize, seen: &mut HashSet<usize>) {
    if let Block::Basic { successors, .. } = &cfg.blocks[id]
        && seen.insert(id)
    {
        for block_id in successors {
            mark_reachable_blocks(cfg, *block_id, seen);
        }
    }
}

/// Removes redundant jump instructions, useless label instructions, and
/// unreachable blocks from the control flow graph (_CFG_).
///
/// Redundant jumps are those that only jump to the next block in sequence.
/// Useless labels are those that are not targeted by any block but the
/// previous.
fn clean_cfg(cfg: &mut CFG<'_>, seen: &HashSet<usize>) {
    let entry_id = 0;
    let exit_id = cfg.blocks.len() - 1;

    // Iterate over the blocks, excluding the entry and exit blocks.
    for i in entry_id + 1..exit_id {
        let prev_id = cfg.blocks[i - 1].id();
        let next_id = cfg.blocks[i + 1].id();
        let block = &mut cfg.blocks[i];

        if let Block::Basic {
            instructions,
            predecessors,
            successors,
            ..
        } = block
        {
            // Retain only the reachable predecessors in the `CFG`.
            predecessors.retain(|id| seen.contains(id));

            if let Some(
                Instruction::Jump(_)
                | Instruction::JumpIfZero { .. }
                | Instruction::JumpIfNotZero { .. },
            ) = instructions.last()
            {
                // Exclude the last basic block since a `jump` instruction at
                // the  end of the graph is never redundant since it targets the
                // `exit` block.
                if next_id == exit_id {
                    continue;
                }

                let mut keep = false;

                for id in successors {
                    // There is a successor block other than the next
                    // block where control can flow to.
                    if *id != next_id {
                        keep = true;
                        break;
                    }
                }

                if !keep {
                    instructions.pop();
                }
            }

            if let Some(Instruction::Label(_)) = instructions.first() {
                let mut keep = false;

                for id in predecessors {
                    // There is a predecessor block other than the previous
                    // block which targets this label.
                    if *id != prev_id {
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

    // Update the predecessors of the exit block, removing references to
    // blocks that have been removed.
    if let Block::Exit { predecessors, .. } = &mut cfg.blocks[exit_id] {
        // Retain only the reachable predecessors in the `CFG`.
        predecessors.retain(|id| seen.contains(id));
    }
}
