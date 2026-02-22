//! Register Allocation (_x86-64_)
//!
//! Handles the assignment of virtual (pseudo) registers to physical registers
//! on the _x86-64_ architecture. This pass ensures efficient use of CPU
//! registers, resolves conflicts, and inserts spills and reloads as needed
//! during code generation.

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;

use crate::compiler::mir::{self, Instruction, Operand, Reg};
use crate::compiler::opt::analysis::run_analysis;
use crate::compiler::opt::targets::x86_64::RegisterLiveness;
use crate::compiler::opt::{Block, CFG, CFGInstruction};
use crate::compiler::parser::sema::symbols::SymbolMap;

/// Type of registers in interference graph.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum RegisterType<'a> {
    Virtual(&'a str),
    Hardware(Reg),
}

impl<'a> TryFrom<Operand<'a>> for RegisterType<'a> {
    type Error = ();

    fn try_from(op: Operand<'a>) -> Result<Self, Self::Error> {
        match op {
            Operand::Register(reg) => Ok(RegisterType::Hardware(reg)),
            Operand::Symbol { ident, is_static } if !is_static => Ok(RegisterType::Virtual(ident)),
            _ => Err(()),
        }
    }
}

/// Register colors (for physical register assignments).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(transparent)]
struct Color(usize);

impl Color {
    /// The number of available hardware registers.
    const K: usize = mir::Reg::ALLOCATABLE_REGS.len();

    /// Returns a new `Color`, or `None` if it does not fit within the range
    /// of available color values.
    #[inline]
    #[must_use]
    fn new(value: usize) -> Self {
        debug_assert!(
            (1..=Color::K).contains(&value),
            "color value '{value}' is out of bounds"
        );

        Color(value)
    }

    /// Returns the inner color value (physical register number).
    #[inline]
    #[must_use]
    const fn value(self) -> usize {
        self.0
    }
}

/// Node of an interference graph (either pseudo or hardware register).
#[derive(Debug)]
struct RegisterNode<'a> {
    id: usize,
    neighbors: Vec<usize>,
    ty: RegisterType<'a>,
    /// Cost of spilling the register to memory.
    spill_cost: f64,
    color: Option<Color>,
    pruned: bool,
}

/// Interference graph used in register allocation.
#[derive(Debug)]
struct InterferenceGraph<'a> {
    nodes: Vec<RegisterNode<'a>>,
}

impl<'a> InterferenceGraph<'a> {
    /// Returns a new, initialized, interference graph from the provided
    /// instructions and symbol map.
    fn from_instructions(instructions: &[Instruction<'a>], sym_map: &SymbolMap) -> Self {
        let mut base = Self::base();

        let mut seen = HashSet::new();

        // Add all encountered virtual registers to the graph.
        for instr in instructions {
            let id = base.nodes.len();

            match instr {
                Instruction::Mov { src, dst }
                | Instruction::Cmp { rhs: src, lhs: dst }
                | Instruction::Binary { rhs: src, dst, .. } => {
                    if let Ok(reg @ RegisterType::Virtual(ident)) = (*src).try_into()
                        && seen.insert(ident)
                    {
                        base.nodes.push(RegisterNode {
                            id,
                            neighbors: vec![],
                            ty: reg,
                            spill_cost: 0.0,
                            color: None,
                            pruned: false,
                        });
                    }

                    if let Ok(reg @ RegisterType::Virtual(ident)) = (*dst).try_into()
                        && seen.insert(ident)
                    {
                        base.nodes.push(RegisterNode {
                            id,
                            neighbors: vec![],
                            ty: reg,
                            spill_cost: 0.0,
                            color: None,
                            pruned: false,
                        });
                    }
                }
                Instruction::Unary { dst: op, .. }
                | Instruction::SetC { dst: op, .. }
                | Instruction::Push(op)
                | Instruction::Idiv(op) => {
                    if let Ok(reg @ RegisterType::Virtual(ident)) = (*op).try_into()
                        && seen.insert(ident)
                    {
                        base.nodes.push(RegisterNode {
                            id,
                            neighbors: vec![],
                            ty: reg,
                            spill_cost: 0.0,
                            color: None,
                            pruned: false,
                        });
                    }
                }
                _ => {}
            }
        }

        base.apply_liveness(instructions, sym_map);

        base
    }

    /// Calculates and updates the spill costs for each register, based on its
    /// usage frequency.
    fn compute_spill_costs(&mut self, instructions: &[Instruction<'_>]) {
        // NOTE: Weight the spill_cost based on if the instruction is within a
        // loop.
        let mut update_spill = |op: Operand<'_>| {
            // NOTE: O(n) time complexity.
            if let Ok(reg @ RegisterType::Virtual(_)) = op.try_into()
                && let Some(node) = self.nodes.iter_mut().find(|n| n.ty == reg)
            {
                node.spill_cost += 1.0;
            }
        };

        for instr in instructions {
            match instr {
                Instruction::Mov { src: val, dst }
                | Instruction::Binary { rhs: val, dst, .. }
                | Instruction::Cmp { rhs: val, lhs: dst } => {
                    update_spill(*val);
                    update_spill(*dst);
                }
                Instruction::Unary { dst: val, .. }
                | Instruction::Idiv(val)
                | Instruction::SetC { dst: val, .. }
                | Instruction::Push(val) => {
                    update_spill(*val);
                }
                _ => {}
            }
        }
    }

    /// Performs graph coloring to assign a physical register (represented by a
    /// color) to each pseudo-register.
    fn assign_registers(&mut self) {
        if let Some(start) = self.nodes.iter().position(|node| !node.pruned) {
            // Index of the next node to prune.
            let mut candidate_idx = None;

            for i in start..self.nodes.len() {
                let node = &self.nodes[i];
                if node.pruned {
                    continue;
                }

                let degree = node
                    .neighbors
                    .iter()
                    .filter(|&&id| !self.nodes[id].pruned)
                    .count();

                if degree < Color::K {
                    candidate_idx = Some(i);
                    break;
                }
            }

            if candidate_idx.is_none() {
                let mut best_metric = f64::INFINITY;

                for i in start..self.nodes.len() {
                    let node = &self.nodes[i];
                    if node.pruned {
                        continue;
                    }

                    let degree = node
                        .neighbors
                        .iter()
                        .filter(|&&id| !self.nodes[id].pruned)
                        .count();

                    if degree != 0 {
                        #[allow(clippy::cast_precision_loss)]
                        let spill_metric = node.spill_cost / degree as f64;

                        if spill_metric < best_metric {
                            best_metric = spill_metric;
                            candidate_idx = Some(i);
                        }
                    }
                }
            }

            let candidate_idx = candidate_idx.expect("candidate should have been found");

            self.nodes[candidate_idx].pruned = true;

            // Color the rest of the graph.
            self.assign_registers();

            // Sets the first `K` bits as available colors.
            let mut available: u32 = (1 << Color::K) - 1;

            for &neighbor in &self.nodes[candidate_idx].neighbors {
                if let Some(color) = &self.nodes[neighbor].color {
                    // Clear the bit at `color.value()`: color is no longer
                    // available.
                    available &= !(1 << (color.value() - 1));
                }
            }

            let candidate_node = &mut self.nodes[candidate_idx];

            if available != 0 {
                if let RegisterType::Hardware(reg) = candidate_node.ty
                    && reg.is_callee_saved()
                {
                    // Callee-saved hardware registers: pick the largest
                    // available color.
                    let leading = available.leading_zeros();
                    candidate_node.color = Some(Color::new((u32::BITS - leading) as usize));
                } else {
                    // Other: pick the smallest available color.
                    let trailing = available.trailing_zeros() as usize;
                    candidate_node.color = Some(Color::new(trailing + 1));
                }

                candidate_node.pruned = false;
            }
        }

        // No unpruned node exists: base case reached.
    }

    /// Returns the base (complete) interference graph, containing all
    /// allocatable hardware registers.
    fn base() -> Self {
        let mut graph = Self {
            nodes: Vec::with_capacity(mir::Reg::ALLOCATABLE_REGS.len()),
        };

        let alloc_len = mir::Reg::ALLOCATABLE_REGS.len();

        // Base graph includes all available hardware registers.
        for reg in mir::Reg::ALLOCATABLE_REGS {
            graph.nodes.push(RegisterNode {
                id: graph.nodes.len(),
                neighbors: Vec::with_capacity(alloc_len - 1),
                ty: RegisterType::Hardware(reg),
                // Cannot be spilled to memory.
                spill_cost: f64::INFINITY,
                color: None,
                pruned: false,
            });
        }

        // Add interference edges (all hardware registers interfere with each
        // other).
        for i in 0..graph.nodes.len() {
            for j in (i + 1)..graph.nodes.len() {
                graph.add_interference(i, j);
            }
        }

        graph
    }

    /// Adds edges to the graph based on the results of the liveness data-flow
    /// analysis.
    fn apply_liveness(&mut self, instructions: &[Instruction<'a>], sym_map: &SymbolMap) {
        let mut cfg = CFG::new();
        cfg.sync(instructions);

        let mut liveness = RegisterLiveness::new(cfg.exit_block_id(), sym_map);

        run_analysis(&cfg, &mut liveness);

        self.update_interference(&cfg, &liveness);
    }

    /// Updates the neighbors of virtual registers based on a liveness analysis.
    fn update_interference<I>(&mut self, cfg: &CFG<I>, liveness: &RegisterLiveness<'_>)
    where
        I: CFGInstruction<Instr = Instruction<'a>>,
    {
        for block in &cfg.basic_blocks() {
            if let Block::Basic {
                id: block_id,
                instructions,
                ..
            } = block
            {
                let mut updated = vec![];

                for (i, instr) in instructions.iter().enumerate() {
                    if let Some(live_regs) = liveness.get_instruction_fact(*block_id, i) {
                        let instr = instr.concrete();

                        instr.find_updated(&mut updated);

                        for &lr in live_regs {
                            // Ensure we don't add an edge between this `src`
                            // and it's `dst`.
                            if let Instruction::Mov { src, .. } = instr
                                && (*src).try_into() == Ok(lr)
                            {
                                continue;
                            }

                            for u in updated
                                .iter()
                                .filter_map(|op| RegisterType::try_from(*op).ok())
                            {
                                // NOTE: O(n) time complexity.
                                if lr != u
                                    && let (Some(lr_node), Some(u_node)) = (
                                        self.nodes.iter().find(|node| node.ty == lr),
                                        self.nodes.iter().find(|node| node.ty == u),
                                    )
                                {
                                    self.add_interference(lr_node.id, u_node.id);
                                }
                            }
                        }

                        updated.clear();
                    }
                }
            }
        }
    }

    /// Adds a undirected edge between two nodes in the interference graph.
    #[inline]
    fn add_interference(&mut self, to: usize, from: usize) {
        let node = &mut self.nodes[from];
        if !node.neighbors.contains(&to) {
            node.neighbors.push(to);
        }

        let node = &mut self.nodes[to];
        if !node.neighbors.contains(&from) {
            node.neighbors.push(from);
        }
    }
}

/// Register mapping from colored virtual registers to hardware registers.
#[derive(Debug)]
struct RegisterMap<'a> {
    /// Mapping from virtual register identifier to hardware register.
    reg_map: HashMap<&'a str, Reg>,
    /// Set of callee-saved registers allocated.
    callee_saved: HashSet<Reg>,
}

impl<'a> RegisterMap<'a> {
    /// Returns a new, initialized, register map from the provided interference
    /// graph.
    fn from_graph(ifg: &InterferenceGraph<'a>) -> Self {
        let mut color_map: HashMap<Option<Color>, Reg> = HashMap::default();

        for node in &ifg.nodes {
            if let RegisterType::Hardware(reg) = node.ty {
                color_map.insert(node.color, reg);
            }
        }

        let mut reg_map = HashMap::default();
        let mut callee_saved = HashSet::default();

        for node in &ifg.nodes {
            if let RegisterType::Virtual(ident) = node.ty
                && let Some(reg) = color_map.get(&node.color)
            {
                reg_map.insert(ident, *reg);

                if reg.is_callee_saved() {
                    callee_saved.insert(*reg);
                }
            }
        }

        Self {
            reg_map,
            callee_saved,
        }
    }
}

/// Transforms the _MIR x86-64_ instructions by assigning virtual registers to
/// physical _x86-64_ registers, returning a set containing callee-saved
/// registers used in allocation.
pub fn allocate_registers(
    instructions: &mut Vec<Instruction<'_>>,
    sym_map: &SymbolMap,
) -> HashSet<Reg> {
    let mut ifg = InterferenceGraph::from_instructions(instructions, sym_map);

    ifg.compute_spill_costs(instructions);
    ifg.assign_registers();

    let rm = RegisterMap::from_graph(&ifg);

    let mut to_remove = Vec::new();

    for (i, instr) in instructions.iter_mut().enumerate() {
        match instr {
            Instruction::Mov { src, dst } => {
                if let Operand::Symbol { ident, .. } = src
                    && let Some(reg) = rm.reg_map.get(ident)
                {
                    *src = Operand::Register(*reg);
                }

                if let Operand::Symbol { ident, .. } = dst
                    && let Some(reg) = rm.reg_map.get(ident)
                {
                    let dst_reg = Operand::Register(*reg);

                    if *src == dst_reg {
                        to_remove.push(i);
                        continue;
                    }

                    *dst = dst_reg;
                }
            }
            Instruction::Unary { dst: op, .. }
            | Instruction::Idiv(op)
            | Instruction::Push(op)
            | Instruction::SetC { dst: op, .. } => {
                if let Operand::Symbol { ident, .. } = op
                    && let Some(reg) = rm.reg_map.get(ident)
                {
                    *op = Operand::Register(*reg);
                }
            }
            Instruction::Binary { rhs, dst: op, .. } | Instruction::Cmp { rhs, lhs: op } => {
                if let Operand::Symbol { ident, .. } = rhs
                    && let Some(reg) = rm.reg_map.get(ident)
                {
                    *rhs = Operand::Register(*reg);
                }

                if let Operand::Symbol { ident, .. } = op
                    && let Some(reg) = rm.reg_map.get(ident)
                {
                    *op = Operand::Register(*reg);
                }
            }
            _ => {}
        }
    }

    // Removing instructions from right-left ensures indicies are not
    // affected by shifting.
    for i in to_remove.drain(..).rev() {
        // NOTE: O(n) time complexity.
        instructions.remove(i);
    }

    rm.callee_saved
}
