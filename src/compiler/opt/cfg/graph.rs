use std::collections::HashMap;

use crate::compiler::ir::{Function, Instruction};
use crate::compiler::opt::cfg::iter::BasicBlocks;

/// Types of blocks in a control-flow graph.
#[derive(Debug, PartialEq, Eq)]
pub enum Block<'a> {
    Entry {
        /// Single block that follows the entry block, as _C_ functions have
        /// only one entry point.
        successor: usize,
    },
    /// Sequences of straight-line code.
    Basic {
        id: usize,
        instructions: Vec<Instruction<'a>>,
        /// Blocks that can execute after.
        successors: Vec<usize>,
        /// Blocks that can execute before.
        predecessors: Vec<usize>,
    },
    Exit {
        id: usize,
        /// Blocks that can execute before the exit block.
        predecessors: Vec<usize>,
    },
}

impl Block<'_> {
    /// Entry block ID in a control-flow graph.
    pub const ENTRY_ID: usize = 0;

    /// Returns the `id` of the current block.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> usize {
        match self {
            Block::Entry { .. } => Block::ENTRY_ID,
            Block::Basic { id, .. } | Block::Exit { id, .. } => *id,
        }
    }
}

/// Control-Flow Graph (_CFG_) for an _IR_ function.
#[derive(Debug)]
pub struct CFG<'a> {
    pub blocks: Vec<Block<'a>>,
    /// Maps each label to its corresponding block ID.
    pub label_map: HashMap<String, usize>,
}

impl Default for CFG<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> CFG<'a> {
    /// Returns a new, empty, control-flow graph.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: Vec::default(),
            label_map: HashMap::default(),
        }
    }

    /// Synchronizes the internal state of the control-flow graph with the
    /// instructions of the provided _IR_ function.
    #[inline]
    pub fn sync(&mut self, f: &Function<'a>) {
        if !self.blocks.is_empty() {
            self.blocks.clear();
            self.label_map.clear();
        }

        self.partition_ir(&f.instructions[..]);
        self.build_control_flow();
    }

    /// Applies the optimized control-flow graph to the provided _IR_ function,
    /// returning `true` if changes were made (indicating further optimizations
    /// are possible).
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

    /// Returns immutable basic blocks of the control-flow graph (excluding
    /// entry and exit blocks).
    #[inline]
    #[must_use]
    pub fn basic_blocks(&self) -> BasicBlocks<'_> {
        BasicBlocks::new(&self.blocks)
    }

    /// Returns mutable basic blocks of the control-flow graph (excluding entry
    /// and exit blocks).
    #[inline]
    #[must_use]
    pub fn basic_blocks_mut(&mut self) -> &mut [Block<'a>] {
        let exit_block_idx = self.blocks.len() - 1;
        &mut self.blocks[Block::ENTRY_ID + 1..exit_block_idx]
    }

    /// Returns the ID of the exit block within the control-flow graph.
    ///
    /// # Panics
    ///
    /// Panics if the control-flow graph is malformed.
    #[inline]
    #[must_use]
    pub fn exit_block_id(&self) -> usize {
        match self.blocks.last() {
            Some(Block::Exit { id, .. }) => *id,
            _ => panic!("malformed control-flow graph: missing exit block"),
        }
    }

    /// Partitions the provided _IR_ instructions into control-flow blocks.
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
                            id: self.blocks.len(),
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
                        id: self.blocks.len(),
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
                id: self.blocks.len(),
                instructions: instructions[chunk_start..].to_vec(),
                successors: vec![],
                predecessors: vec![],
            });
        }

        self.blocks.push(Block::Exit {
            id: self.blocks.len(),
            predecessors: vec![],
        });
    }

    /// Builds the control-flow of the graph, linking blocks according to their
    /// instructions and resolving all successors and predecessors.
    fn build_control_flow(&mut self) {
        /// Adds a directed edge between two blocks in the control-flow graph.
        fn add_edge(blocks: &mut [Block<'_>], from: usize, to: usize) {
            // Add `to` as a successor of `from`
            match &mut blocks[from] {
                Block::Entry { successor, .. } => *successor = to,
                Block::Basic { successors, .. } => {
                    successors.push(to);
                }
                Block::Exit { .. } => panic!("exit block should not have successors"),
            }

            // Add `from` as a predecessor of `to`
            match &mut blocks[to] {
                Block::Exit { predecessors, .. } | Block::Basic { predecessors, .. } => {
                    predecessors.push(from);
                }
                Block::Entry { .. } => panic!("entry block should not have predecessors"),
            }
        }

        // When building control-flow, the ID of the exit block is the same as
        // it's index.
        let exit_id = self.exit_block_id();

        // Add an edge from `entry` to the first basic block.
        //
        // Assuming there is at least one basic block, since empty functions are
        // not optimized.
        add_edge(&mut self.blocks, Block::ENTRY_ID, 1);

        // Iterate over the blocks, excluding the entry and exit blocks.
        for block_id in Block::ENTRY_ID + 1..exit_id {
            let next_block_id = block_id + 1;
            let block = &mut self.blocks[block_id];

            if let Block::Basic { instructions, .. } = block {
                // Last instruction of the block determines the control flow.
                if let Some(last) = instructions.last() {
                    match last {
                        Instruction::Return(_) => add_edge(&mut self.blocks, block_id, exit_id),
                        Instruction::Jump(target)
                        | Instruction::JumpIfZero { target, .. }
                        | Instruction::JumpIfNotZero { target, .. } => {
                            let is_unconditional = matches!(last, Instruction::Jump(_));

                            if let Some(target_id) = self.label_map.get(target.as_str()) {
                                add_edge(&mut self.blocks, block_id, *target_id);

                                if !is_unconditional {
                                    add_edge(&mut self.blocks, block_id, next_block_id);
                                }
                            } else {
                                // The `target` block was optimized away. Since
                                // there is no edge to create, ensure control
                                // can fallthrough to the next block.
                                add_edge(&mut self.blocks, block_id, next_block_id);
                            }
                        }
                        _ => add_edge(&mut self.blocks, block_id, next_block_id),
                    }
                }
            }
        }
    }

    /// Converts the control-flow graph back into a list of _IR_ instructions.
    fn cfg_to_ir(&mut self) -> Vec<Instruction<'a>> {
        let mut ir_instructions = vec![];

        for block in self.basic_blocks_mut() {
            if let Block::Basic { instructions, .. } = block
                && !instructions.is_empty()
            {
                ir_instructions.append(instructions);
            }
        }

        ir_instructions
    }
}
