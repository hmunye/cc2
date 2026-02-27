//! Register Allocation (_x86-64_)
//!
//! Handles the assignment of virtual registers to _x86-64_ hardware registers,
//! ensuring efficient use of CPU registers, resolving conflicts, and inserting
//! spills and reloads as needed during code generation.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;

use crate::compiler;
use crate::compiler::frontend::SymbolTable;
use crate::compiler::opt::targets::x86_64::liveness;
use crate::compiler::targets::x86_64::{self, Instruction, Operand, Reg};

/// Type of registers.
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

/// Hardware register color as a newtype wrapper.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(transparent)]
pub struct Color(usize);

impl Color {
    /// Number of available hardware registers.
    pub const K: usize = x86_64::Reg::ALLOCATABLE_REGS.len();

    /// Returns a new `Color`.
    #[inline]
    #[must_use]
    fn new(value: usize) -> Self {
        debug_assert!(
            (1..=Color::K).contains(&value),
            "color value '{value}' is out of bounds"
        );

        Color(value)
    }

    /// Returns the inner color value.
    #[inline]
    #[must_use]
    const fn to_value(self) -> usize {
        self.0
    }
}

/// Node of an interference graph.
#[derive(Debug)]
pub struct RegisterNode<'a> {
    pub id: usize,
    pub neighbors: HashSet<usize>,
    pub ty: RegisterType<'a>,
    /// Cost of spilling the register to memory.
    spill_cost: f64,
    color: Option<Color>,
    pruned: bool,
}

/// Interference graph used in register allocation.
#[derive(Debug)]
pub struct InterferenceGraph<'a> {
    pub nodes: Vec<Option<RegisterNode<'a>>>,
    pub ty_to_index: HashMap<RegisterType<'a>, usize>,
}

impl<'a> InterferenceGraph<'a> {
    /// Returns a new, initialized, interference graph from the provided
    /// instructions.
    #[inline]
    #[must_use]
    pub fn from_instructions(instructions: &[Instruction<'a>], sym_table: &'a SymbolTable) -> Self {
        let mut base = Self::base();

        base.insert_virtual_registers(instructions);

        liveness::update_interference(&mut base, instructions, sym_table);

        base
    }

    /// Adds a undirected edge between two nodes in the graph.
    ///
    /// # Panics
    ///
    /// Panics if the provided indices are out-of-bounds.
    #[inline]
    pub fn add_interference(&mut self, to: usize, from: usize) {
        if let Some(node) = &mut self.nodes[to] {
            node.neighbors.insert(from);
        }

        if let Some(node) = &mut self.nodes[from] {
            node.neighbors.insert(to);
        }
    }

    /// Removes an undirected edge between two nodes in the graph.
    ///
    /// # Panics
    ///
    /// Panics if the provided indices are out-of-bounds.
    #[inline]
    pub fn remove_interference(&mut self, to: usize, from: usize) {
        if let Some(node) = &mut self.nodes[to] {
            node.neighbors.remove(&from);
        }

        if let Some(node) = &mut self.nodes[from] {
            node.neighbors.remove(&to);
        }
    }

    /// Returns `true` if the nodes `to` and `from` interfere.
    ///
    /// # Panics
    ///
    /// Panics if the provided indices are out-of-bounds.
    #[inline]
    #[must_use]
    pub fn are_neighbors(&self, to: usize, from: usize) -> bool {
        if let (Some(to_node), Some(from_node)) =
            (self.nodes[to].as_ref(), self.nodes[from].as_ref())
        {
            to_node.neighbors.contains(&from) || from_node.neighbors.contains(&to)
        } else {
            false
        }
    }

    /// Removes the node `remove` from the graph, transferring all neighbors
    /// to the node `keep`.
    ///
    /// # Panics
    ///
    /// Panics if the provided indices are out-of-bounds.
    #[inline]
    pub fn remove_node(&mut self, remove: usize, keep: usize) {
        if let Some(node) = self.nodes[remove].take() {
            self.ty_to_index.remove(&node.ty);

            for id in node.neighbors {
                self.add_interference(keep, id);
                self.remove_interference(remove, id);
            }
        }
    }

    /// Returns the base (complete) interference graph, containing all
    /// allocatable hardware registers.
    fn base() -> Self {
        let alloc_len = x86_64::Reg::ALLOCATABLE_REGS.len();

        let mut graph = Self {
            nodes: Vec::with_capacity(alloc_len),
            ty_to_index: HashMap::with_capacity(alloc_len),
        };

        // Base graph includes all available hardware registers.
        for reg in x86_64::Reg::ALLOCATABLE_REGS {
            let ty = RegisterType::Hardware(reg);
            let id = graph.nodes.len();

            graph.nodes.push(Some(RegisterNode {
                id,
                // Not including itself.
                neighbors: HashSet::with_capacity(alloc_len - 1),
                ty,
                // Cannot be spilled to memory.
                spill_cost: f64::INFINITY,
                color: None,
                pruned: false,
            }));

            graph.ty_to_index.insert(ty, id);
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

    /// Inserts all virtual registers within the provided instructions to the
    /// graph.
    fn insert_virtual_registers(&mut self, instructions: &[Instruction<'a>]) {
        for instr in instructions {
            let id = self.nodes.len();

            match instr {
                Instruction::Mov { src, dst }
                | Instruction::Cmp { rhs: src, lhs: dst }
                | Instruction::Binary { rhs: src, dst, .. } => {
                    if let Ok(reg @ RegisterType::Virtual(_)) = (*src).try_into()
                        && let Entry::Vacant(entry) = self.ty_to_index.entry(reg)
                    {
                        entry.insert(id);

                        self.nodes.push(Some(RegisterNode {
                            id,
                            neighbors: HashSet::default(),
                            ty: reg,
                            spill_cost: 0.0,
                            color: None,
                            pruned: false,
                        }));
                    }

                    if let Ok(reg @ RegisterType::Virtual(_)) = (*dst).try_into()
                        && let Entry::Vacant(entry) = self.ty_to_index.entry(reg)
                    {
                        entry.insert(id);

                        self.nodes.push(Some(RegisterNode {
                            id,
                            neighbors: HashSet::default(),
                            ty: reg,
                            spill_cost: 0.0,
                            color: None,
                            pruned: false,
                        }));
                    }
                }
                Instruction::Unary { dst: op, .. }
                | Instruction::SetC { dst: op, .. }
                | Instruction::Push(op)
                | Instruction::Idiv(op) => {
                    if let Ok(reg @ RegisterType::Virtual(_)) = (*op).try_into()
                        && let Entry::Vacant(entry) = self.ty_to_index.entry(reg)
                    {
                        entry.insert(id);

                        self.nodes.push(Some(RegisterNode {
                            id,
                            neighbors: HashSet::default(),
                            ty: reg,
                            spill_cost: 0.0,
                            color: None,
                            pruned: false,
                        }));
                    }
                }
                _ => {}
            }
        }
    }

    /// Computes the spill costs for each register within the graph, based on
    /// usage frequency.
    fn compute_spill_costs(&mut self, instructions: &[Instruction<'_>]) {
        // TODO: Weigh the `spill_cost` based on if the instruction is within a
        // loop.
        let mut update_spill = |op: Operand<'_>| {
            if let Ok(reg @ RegisterType::Virtual(_)) = op.try_into()
                && let Some(&idx) = self.ty_to_index.get(&reg)
                && let Some(node) = &mut self.nodes[idx]
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

    /// Performs graph coloring, assigning a hardware register (represented by a
    /// color) to each virtual register.
    fn color_registers(&mut self) {
        if let Some(start) = self
            .nodes
            .iter()
            .position(|node| node.as_ref().is_some_and(|n| !n.pruned))
        {
            // Index of the next node to prune.
            let mut candidate_idx = None;

            for i in start..self.nodes.len() {
                if let Some(node) = &self.nodes[i] {
                    if node.pruned {
                        continue;
                    }

                    let degree = node
                        .neighbors
                        .iter()
                        .filter(|&&id| !self.nodes[id].as_ref().is_some_and(|n| n.pruned))
                        .count();

                    if degree < Color::K {
                        candidate_idx = Some(i);
                        break;
                    }
                }
            }

            if candidate_idx.is_none() {
                let mut best_metric = f64::INFINITY;

                for i in start..self.nodes.len() {
                    if let Some(node) = &self.nodes[i] {
                        if node.pruned {
                            continue;
                        }

                        let degree = node
                            .neighbors
                            .iter()
                            .filter(|&&id| !self.nodes[id].as_ref().is_some_and(|n| n.pruned))
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
            }

            let candidate_idx = candidate_idx.expect("candidate node index should have been found");

            if let Some(candidate) = self.nodes[candidate_idx].as_mut() {
                candidate.pruned = true;
            }

            self.color_registers();

            // Sets the first `K` color bits as available.
            let mut available: u32 = (1 << Color::K) - 1;

            if let Some(neighbors) = self.nodes[candidate_idx].as_ref().map(|n| &n.neighbors) {
                for &id in neighbors {
                    if let Some(color) =
                        self.nodes[id].as_ref().and_then(|node| node.color.as_ref())
                    {
                        // Clear the bit: color is no longer available.
                        available &= !(1 << (color.to_value() - 1));
                    }
                }
            }

            if let Some(candidate) = self.nodes[candidate_idx].as_mut()
                && available != 0
            {
                if let RegisterType::Hardware(reg) = candidate.ty
                    && reg.is_callee_saved()
                {
                    // Pick the largest available color.
                    let leading = available.leading_zeros();

                    candidate.color = Some(Color::new((u32::BITS - leading) as usize));
                } else {
                    // Pick the smallest available color.
                    let trailing = available.trailing_zeros() as usize;

                    candidate.color = Some(Color::new(trailing + 1));
                }

                candidate.pruned = false;
            }
        }

        // No unpruned node exists: base case reached.
    }
}

/// Register mapping from colored virtual registers to hardware registers.
#[derive(Debug)]
struct RegisterMap<'a> {
    /// Mapping from virtual register identifiers to hardware registers.
    reg_map: HashMap<&'a str, Reg>,
    /// Set of callee-saved registers used during allocation.
    callee_saved: HashSet<Reg>,
}

impl<'a> RegisterMap<'a> {
    /// Returns a new, initialized, register map from the provided interference
    /// graph.
    fn from_graph(ifg: InterferenceGraph<'a>) -> Self {
        let mut color_map: HashMap<Option<Color>, Reg> =
            HashMap::with_capacity(x86_64::Reg::ALLOCATABLE_REGS.len());

        for node in &ifg.nodes {
            if let Some(node) = node
                && let RegisterType::Hardware(reg) = node.ty
            {
                color_map.insert(node.color, reg);
            }
        }

        let mut reg_map = HashMap::default();
        let mut callee_saved = HashSet::default();

        for node in ifg.nodes {
            if let Some(node) = node
                && let RegisterType::Virtual(ident) = node.ty
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

/// Transforms the provided _MIR x86-64_ instructions, assigning virtual
/// registers to hardware registers, returning a `HashSet` containing
/// callee-saved registers used during allocation.
pub fn allocate_registers<'a>(
    instructions: &mut Vec<Instruction<'a>>,
    sym_table: &'a SymbolTable,
    coalesce: bool,
) -> HashSet<Reg> {
    let mut ifg = if coalesce {
        compiler::opt::targets::x86_64::coalesce_loop(instructions, sym_table)
    } else {
        InterferenceGraph::from_instructions(instructions, sym_table)
    };

    ifg.compute_spill_costs(instructions);
    ifg.color_registers();

    let rm = RegisterMap::from_graph(ifg);

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

    // Removing instructions from right-left ensures indicies are not affected
    // by shifting.
    for i in to_remove.drain(..).rev() {
        instructions.remove(i);
    }

    rm.callee_saved
}
