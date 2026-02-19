use std::collections::HashMap;

use crate::compiler::ir;
use crate::compiler::opt::cfg::iter::BasicBlocks;

/// Types of instructions that determine a basic block boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockBoundary<'a> {
    /// Ends the block and returns from the function.
    Return,
    /// Ends the block and jumps unconditionally to a target.
    Jump(&'a str),
    /// Ends the block and conditionally jumps to a target.
    CondJump(&'a str),
    /// Ends the current block and starts a new one.
    Label(&'a str),
    /// Does not affect control flow or block boundaries.
    Other,
}

/// Trait for instructions that expose control-flow behavior, used in a
/// control-flow graph.
pub trait CFGInstruction {
    /// Concrete type of the instruction.
    type Instr;

    /// Returns the block boundary classification of this instruction.
    #[must_use]
    fn block_boundary(&self) -> BlockBoundary<'_>;

    /// Returns a reference to the underlying concrete instruction type.
    #[must_use]
    fn concrete(&self) -> &Self::Instr;

    /// Returns a mutable reference to the underlying concrete instruction type.
    #[must_use]
    fn concrete_mut(&mut self) -> &mut Self::Instr;
}

impl<'a> CFGInstruction for ir::Instruction<'a> {
    type Instr = ir::Instruction<'a>;

    #[inline]
    fn block_boundary(&self) -> BlockBoundary<'_> {
        match self {
            ir::Instruction::Return(_) => BlockBoundary::Return,
            ir::Instruction::Jump(target) => BlockBoundary::Jump(target),
            ir::Instruction::JumpIfZero { target, .. }
            | ir::Instruction::JumpIfNotZero { target, .. } => BlockBoundary::CondJump(target),
            ir::Instruction::Label(label) => BlockBoundary::Label(label),
            _ => BlockBoundary::Other,
        }
    }

    #[inline]
    fn concrete(&self) -> &Self::Instr {
        self
    }

    #[inline]
    fn concrete_mut(&mut self) -> &mut Self::Instr {
        self
    }
}

/// Types of blocks in a control-flow graph.
#[derive(Debug, PartialEq, Eq)]
pub enum Block<I> {
    Entry {
        /// Single block that follows the entry block, as _C_ functions have
        /// only one entry point.
        successor: usize,
    },
    /// Sequences of straight-line code.
    Basic {
        id: usize,
        instructions: Vec<I>,
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

impl<I> Block<I> {
    /// Entry ID in any control-flow graph.
    pub const ENTRY_ID: usize = 0;

    /// Returns the `id` of the current block.
    #[inline]
    #[must_use]
    pub const fn id(&self) -> usize {
        match self {
            Block::Entry { .. } => Block::<I>::ENTRY_ID,
            Block::Basic { id, .. } | Block::Exit { id, .. } => *id,
        }
    }
}

/// Control-Flow Graph (_CFG_).
#[derive(Debug)]
pub struct CFG<I> {
    pub blocks: Vec<Block<I>>,
    /// Maps each label to its corresponding block ID.
    pub label_map: HashMap<String, usize>,
}

impl<I: CFGInstruction> Default for CFG<I> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I: CFGInstruction> CFG<I> {
    /// Returns a new, empty, control-flow graph.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            blocks: Vec::default(),
            label_map: HashMap::default(),
        }
    }

    /// Returns immutable basic blocks of the control-flow graph (excluding
    /// entry and exit blocks).
    #[inline]
    #[must_use]
    pub fn basic_blocks(&self) -> BasicBlocks<'_, I> {
        BasicBlocks::new(&self.blocks)
    }

    /// Returns mutable basic blocks of the control-flow graph (excluding entry
    /// and exit blocks).
    #[inline]
    #[must_use]
    pub fn basic_blocks_mut(&mut self) -> &mut [Block<I>] {
        let exit_block_idx = self.blocks.len() - 1;
        &mut self.blocks[Block::<I>::ENTRY_ID + 1..exit_block_idx]
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

    /// Builds the control-flow of the graph, linking blocks according to their
    /// instructions and resolving all successors and predecessors.
    fn build_control_flow(&mut self) {
        /// Adds a directed edge between two blocks in the control-flow graph.
        fn add_edge<I: CFGInstruction>(blocks: &mut [Block<I>], from: usize, to: usize) {
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
        let entry_id = Block::<I>::ENTRY_ID;

        // Add an edge from `entry` to the first basic block.
        //
        // Assuming there is at least one basic block, since empty functions are
        // not optimized.
        add_edge(&mut self.blocks, entry_id, 1);

        // Iterate over the blocks, excluding the entry and exit blocks.
        for block_id in entry_id + 1..exit_id {
            let next_block_id = block_id + 1;
            let block = &mut self.blocks[block_id];

            if let Block::Basic { instructions, .. } = block {
                // Last instruction of the block determines the control flow.
                if let Some(last) = instructions.last() {
                    let boundary = last.block_boundary();
                    match boundary {
                        BlockBoundary::Return => add_edge(&mut self.blocks, block_id, exit_id),
                        BlockBoundary::Jump(target) | BlockBoundary::CondJump(target) => {
                            let is_unconditional = matches!(boundary, BlockBoundary::Jump(_));

                            if let Some(target_id) = self.label_map.get(target) {
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
}

impl<I: CFGInstruction + Clone> CFG<I> {
    /// Synchronizes the internal state of the control-flow graph with the
    /// provided instructions.
    #[inline]
    pub fn sync(&mut self, instructions: &[I]) {
        if !self.blocks.is_empty() {
            self.blocks.clear();
            self.label_map.clear();
        }

        self.partition(instructions);
        self.build_control_flow();
    }

    /// Partitions the provided instructions into control-flow blocks.
    fn partition(&mut self, instructions: &[I]) {
        self.blocks.push(Block::Entry {
            // Sentinel value.
            successor: usize::MAX,
        });

        // Tracks the start of an instruction chunk range.
        let mut chunk_start = 0;

        for i in 0..instructions.len() {
            let inst = &instructions[i];

            match inst.block_boundary() {
                BlockBoundary::Label(label) => {
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
                    self.label_map.insert(label.to_string(), self.blocks.len());
                }
                BlockBoundary::Jump(_) | BlockBoundary::CondJump(_) | BlockBoundary::Return => {
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
                BlockBoundary::Other => {
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
}

impl<I: CFGInstruction + PartialEq> CFG<I> {
    /// Applies the optimized control-flow graph to the provided instructions,
    /// returning `true` if changes were made (indicating further optimizations
    /// are possible).
    #[inline]
    #[must_use]
    pub fn apply(&mut self, instructions: &mut Vec<I>) -> bool {
        let mut optimized_cfg = self.flatten_cfg();

        let is_changed = optimized_cfg != *instructions;

        if is_changed {
            std::mem::swap(instructions, &mut optimized_cfg);
        }

        is_changed
    }

    /// Flattens the control-flow graph into a list of instructions.
    fn flatten_cfg(&mut self) -> Vec<I> {
        let mut opt_instructions = vec![];

        for block in self.basic_blocks_mut() {
            if let Block::Basic { instructions, .. } = block
                && !instructions.is_empty()
            {
                opt_instructions.append(instructions);
            }
        }

        opt_instructions
    }
}
