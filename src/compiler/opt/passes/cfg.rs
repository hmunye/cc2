//! Control Flow Graph
//!
//! Constructs and analyzes the control flow graph (CFG) for functions in an
//! intermediate representation (_IR_), representing basic blocks and control
//! flow edges, used for intraprocedural optimizations and analysis.

use std::collections::HashMap;

use crate::compiler::ir::{Function, Instruction};

/// Types of blocks in a control flow graph (_CFG_).
#[derive(Debug)]
enum Block<'a> {
    Entry {
        /// Single block that follows the entry block, as _C_ functions have
        /// only one entry point.
        successor: usize,
    },
    /// Sequences of straight-line code.
    Basic {
        instructions: Vec<Instruction<'a>>,
        /// Blocks that can execute after.
        successors: Vec<usize>,
        /// Blocks that can execute before.
        predecessors: Vec<usize>,
    },
    Exit {
        /// Blocks that can execute before the exit block.
        predecessors: Vec<usize>,
    },
}

/// Control Flow Graph (_CFG_) for a given _IR_ function.
#[derive(Debug)]
pub struct CFG<'a> {
    blocks: Vec<Block<'a>>,
    /// Maps each label to its corresponding block ID.
    label_map: HashMap<String, usize>,
}

impl Default for CFG<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CFG<'a> {
    /// Returns a new `CFG`.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: Vec::default(),
            label_map: HashMap::default(),
        }
    }

    /// Synchronizes the internal state of the control flow graph (_CFG_) with
    /// the current state of the provided _IR_ function.
    #[inline]
    pub fn sync(&mut self, f: &Function<'a>) {
        self.blocks.clear();
        self.label_map.clear();

        self.partition_ir(&f.instructions[..]);
        self.build_control_flow();
    }

    /// Applies optimizations to the _IR_ function using the optimized control
    /// flow graph. Returns `true` if changes were made, indicating further
    /// optimizations are possible.
    #[inline]
    #[must_use]
    pub fn apply(&mut self, f: &mut Function<'a>) -> bool {
        let mut optimized_ir = self.cfg_to_ir();

        let is_changed = optimized_ir != f.instructions;

        if is_changed {
            std::mem::swap(&mut f.instructions, &mut optimized_ir);
        }

        is_changed
    }

    /// Partitions the provided _IR_ instructions into _CFG_ blocks.
    fn partition_ir(&mut self, instructions: &[Instruction<'a>]) {
        self.blocks.push(Block::Entry {
            // Sentinel value.
            successor: usize::MAX,
        });

        // Tracks the start of an instruction chunk range.
        let mut chunk_start = 0;

        for i in 0..instructions.len() {
            let inst = &instructions[i];

            match inst {
                Instruction::Label(label) => {
                    // Finalize the previous block if there are instructions
                    // before the label.
                    if i > chunk_start {
                        self.blocks.push(Block::Basic {
                            instructions: instructions[chunk_start..i].to_vec(),
                            successors: vec![],
                            predecessors: vec![],
                        });
                    }

                    // Include the current index as part of the next instruction
                    // chunk range.
                    chunk_start = i;

                    // The current label is associated with the next block ID.
                    self.label_map.insert(label.clone(), self.blocks.len());
                }
                Instruction::Jump(_)
                | Instruction::JumpIfZero { .. }
                | Instruction::JumpIfNotZero { .. }
                | Instruction::Return(_) => {
                    self.blocks.push(Block::Basic {
                        instructions: instructions[chunk_start..=i].to_vec(),
                        successors: vec![],
                        predecessors: vec![],
                    });

                    // Exclude the current index from being part of the next
                    // instruction chunk range.
                    chunk_start = i + 1;
                }
                _ => {
                    // Allow the range from `chunk_start..i` to continue
                    // growing.
                }
            }
        }

        if chunk_start < instructions.len() {
            self.blocks.push(Block::Basic {
                instructions: instructions[chunk_start..].to_vec(),
                successors: vec![],
                predecessors: vec![],
            });
        }

        self.blocks.push(Block::Exit {
            predecessors: vec![],
        });
    }

    /// Builds the control flow by linking blocks according to control flow
    /// instructions, resolving block successors and predecessors.
    fn build_control_flow(&mut self) {
        /// Adds a directed edge between two blocks in the control flow graph.
        fn add_edge(blocks: &mut [Block<'_>], from: usize, to: usize) {
            // Add `to` as a successor of `from`
            match &mut blocks[from] {
                Block::Entry { successor } => *successor = to,
                Block::Basic { successors, .. } => {
                    successors.push(to);
                }
                Block::Exit { .. } => (),
            }

            // Add `from` as a predecessor of `to`
            match &mut blocks[to] {
                Block::Exit { predecessors } | Block::Basic { predecessors, .. } => {
                    predecessors.push(from);
                }
                Block::Entry { .. } => (),
            }
        }

        let entry_id = 0;
        let exit_id = self.blocks.len() - 1;

        // Add an edge from `entry` to the first basic block. Assuming there
        // is at least one basic block, since empty functions are not optimized.
        add_edge(&mut self.blocks, entry_id, 1);

        // Iterate over the blocks, excluding the entry and exit blocks.
        for block_id in entry_id + 1..exit_id {
            let block = &mut self.blocks[block_id];

            if let Block::Basic { instructions, .. } = block {
                // Last instruction of the block determines the control flow.
                if let Some(last) = instructions.last() {
                    let next_block_id = block_id + 1;

                    match last {
                        Instruction::Return(_) => add_edge(&mut self.blocks, block_id, exit_id),
                        Instruction::Jump(target)
                        | Instruction::JumpIfZero { target, .. }
                        | Instruction::JumpIfNotZero { target, .. } => {
                            let target_id = self.label_map.get(target.as_str()).expect(
                                "all labels should be mapped to a block ID during partitioning",
                            );

                            let is_unconditional = matches!(last, Instruction::Jump(_));

                            add_edge(&mut self.blocks, block_id, *target_id);

                            if !is_unconditional {
                                add_edge(&mut self.blocks, block_id, next_block_id);
                            }
                        }
                        _ => add_edge(&mut self.blocks, block_id, next_block_id),
                    }
                }
            }
        }
    }

    /// Converts the control flow graph (_CFG_) back into a list of _IR_
    /// instructions, leaving each block's instructions empty.
    fn cfg_to_ir(&mut self) -> Vec<Instruction<'a>> {
        let entry_id = 0;
        let exit_id = self.blocks.len() - 1;

        let mut ir_instructions = vec![];

        // Iterate over the blocks, excluding the entry and exit blocks.
        for block in &mut self.blocks[entry_id + 1..exit_id] {
            if let Block::Basic { instructions, .. } = block
                && !instructions.is_empty()
            {
                ir_instructions.append(instructions);
            }
        }

        ir_instructions
    }
}
