//! Register Coalescing (_x86-64_)
//!
//! Optimizes the use of _x86-64_ hardware registers by coalescing (merging)
//! compatible virtual registers. This pass attempts to reduce the number of
//! registers required by eliminating redundant moves, improving register usage
//! efficiency, and minimizing instruction count during code generation.

use std::collections::{HashMap, HashSet};

use crate::compiler::frontend::SymbolTable;
use crate::compiler::opt::targets::x86_64::RegisterType;
use crate::compiler::opt::targets::x86_64::register_alloc::{
    Color, InterferenceGraph, RegisterNode,
};
use crate::compiler::targets::x86_64::{Instruction, Operand};

/// A disjoint set (union-find) data structure for managing a collection of
/// disjoint sets of elements.
#[derive(Debug)]
struct DisjointSet<T> {
    sets: HashMap<T, T>,
}

impl<T> DisjointSet<T> {
    /// Returns a new, empty, `DisjointSet`.
    #[inline]
    fn new() -> Self {
        Self {
            sets: HashMap::new(),
        }
    }

    /// Returns `true` if the disjoint set contains no elements.
    #[inline]
    fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }
}

impl<T: Copy + Eq + std::hash::Hash> DisjointSet<T> {
    /// Union (merge) the sets containing `x` and `y`, setting `x` as the set
    /// representative.
    #[inline]
    fn union(&mut self, x: T, y: T) {
        // Ensure both sets are present.
        self.sets.entry(x).or_insert(x);
        self.sets.entry(y).or_insert(y);

        let root_x = self.find(x);
        let root_y = self.find(y);

        // Check if both sets are already merged.
        if root_x != root_y {
            // Update the representative of the set containing `root_y`.
            self.sets.insert(root_y, root_x);
        }
    }

    /// Returns the representative of the set containing `x`.
    #[inline]
    fn find(&mut self, x: T) -> T {
        if let Some(&rep) = self.sets.get(&x)
            && rep != x
        {
            let root = self.find(rep);
            // Apply `path compression`: make each element along the path point
            // directly to it's root.
            self.sets.insert(x, root);
            root
        } else {
            x
        }
    }
}

/// Coalesces register assignments in a set of instructions, iteratively
/// merging registers until no further coalescing can be done.
#[must_use]
pub fn coalesce_loop<'a>(
    instructions: &mut Vec<Instruction<'a>>,
    sym_table: &'a SymbolTable,
) -> InterferenceGraph<'a> {
    let mut ifg = InterferenceGraph::from_instructions(instructions, sym_table);

    loop {
        if !coalesce_registers(&mut ifg, instructions) {
            break;
        }

        ifg = InterferenceGraph::from_instructions(instructions, sym_table);
    }

    ifg
}

/// Transforms the provided _MIR x86-64_ instructions by coalescing (merging)
/// compatible virtual registers, returning `true` if coalescing occurred
/// (indicating further coalescing is possible).
pub fn coalesce_registers<'a>(
    ifg: &mut InterferenceGraph<'a>,
    instructions: &mut Vec<Instruction<'a>>,
) -> bool {
    let mut coalesced = DisjointSet::new();

    for instr in instructions.iter() {
        if let Instruction::Mov { src, dst } = instr
            && let Ok(src_reg) = (*src).try_into()
            && let Ok(dst_reg) = (*dst).try_into()
        {
            let src = coalesced.find(src_reg);
            let dst = coalesced.find(dst_reg);

            if src == dst {
                continue;
            }

            // NOTE: Linearly scan `nodes` instead of using `ty_to_index` since
            // the graph is mutated in-place during coalescing (`remove_node`).
            let (s_node, d_node) = {
                let mut src_node = None;
                let mut dst_node = None;

                for n in ifg.nodes.iter().flatten() {
                    match n.ty {
                        t if t == src => src_node = Some(n),
                        t if t == dst => dst_node = Some(n),
                        _ => {}
                    }

                    if src_node.is_some() && dst_node.is_some() {
                        break;
                    }
                }

                (src_node, dst_node)
            };

            if let (Some(src), Some(dst)) = (s_node, d_node)
                && !ifg.are_neighbors(src.id, dst.id)
                && is_conservative(ifg, src, dst)
            {
                let (keep, merge) = {
                    if matches!(src.ty, RegisterType::Hardware(_)) {
                        (src, dst)
                    } else {
                        (dst, src)
                    }
                };

                coalesced.union(keep.ty, merge.ty);
                ifg.remove_node(merge.id, keep.id);
            }
        }
    }

    let was_coalesced = !coalesced.is_empty();

    if was_coalesced {
        rewrite_coalesced(instructions, &mut coalesced);
    }

    was_coalesced
}

fn rewrite_coalesced<'a>(
    instructions: &mut Vec<Instruction<'a>>,
    coalesced: &mut DisjointSet<RegisterType<'a>>,
) {
    let mut coalesce_op = |op: &mut Operand<'a>| {
        (*op).try_into().map_or_else(
            |()| *op,
            |r| match coalesced.find(r) {
                RegisterType::Hardware(reg) => Operand::Register(reg),
                RegisterType::Virtual(ident) => Operand::Symbol {
                    ident,
                    is_static: false,
                },
            },
        )
    };

    let mut to_remove = Vec::new();

    for (i, inst) in instructions.iter_mut().enumerate() {
        match inst {
            Instruction::Mov { src, dst } => {
                let c_src = coalesce_op(src);
                let c_dst = coalesce_op(dst);

                if c_src == c_dst {
                    to_remove.push(i);
                    continue;
                }

                *src = c_src;
                *dst = c_dst;
            }
            Instruction::Unary { dst: val, .. }
            | Instruction::Idiv(val)
            | Instruction::Push(val)
            | Instruction::SetC { dst: val, .. } => {
                *val = coalesce_op(val);
            }
            Instruction::Binary { rhs, dst: val, .. } | Instruction::Cmp { rhs, lhs: val } => {
                *rhs = coalesce_op(rhs);
                *val = coalesce_op(val);
            }
            _ => {}
        }
    }

    // Removing instructions from right-left ensures indicies are not affected
    // by shifting.
    for i in to_remove.into_iter().rev() {
        instructions.remove(i);
    }
}

/// Returns `true` if the provided register nodes can be coalesced
/// conservatively (will not make graph coloring more costly).
fn is_conservative(
    ifg: &InterferenceGraph<'_>,
    src: &RegisterNode<'_>,
    dst: &RegisterNode<'_>,
) -> bool {
    fn briggs_test(
        ifg: &InterferenceGraph<'_>,
        src: &RegisterNode<'_>,
        dst: &RegisterNode<'_>,
    ) -> bool {
        let mut significant_n = 0;

        let src_id = src.id;
        let dst_id = dst.id;

        let combined_neighbors = src
            .neighbors
            .iter()
            .copied()
            .chain(dst.neighbors.iter().copied())
            .collect::<HashSet<_>>();

        for id in combined_neighbors {
            if let Some(neighbor) = &ifg.nodes[id] {
                let mut degree = neighbor.neighbors.len();

                if ifg.are_neighbors(id, src_id) && ifg.are_neighbors(id, dst_id) {
                    degree = degree.saturating_sub(1);
                }

                if degree >= Color::K {
                    significant_n += 1;
                }
            }
        }

        significant_n < Color::K
    }

    fn george_test(
        ifg: &InterferenceGraph<'_>,
        hardware: &RegisterNode<'_>,
        virt: &RegisterNode<'_>,
    ) -> bool {
        let hard_id = hardware.id;

        for &id in &virt.neighbors {
            if ifg.are_neighbors(id, hard_id) {
                continue;
            }

            if let Some(neighbor) = &ifg.nodes[id]
                && neighbor.neighbors.len() < Color::K
            {
                continue;
            }

            return false;
        }

        true
    }

    if briggs_test(ifg, src, dst) {
        return true;
    }

    match (src.ty, dst.ty) {
        (RegisterType::Hardware(_), _) => george_test(ifg, src, dst),
        (_, RegisterType::Hardware(_)) => george_test(ifg, dst, src),
        _ => false,
    }
}
