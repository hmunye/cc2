//! Optimization Pipeline
//!
//! Executes machine-independent optimization passes on an intermediate
//! representation (_IR_) based on user-specified options.

use crate::args::Opts;
use crate::compiler::opt::passes::cfg::CFG;
use crate::compiler::{
    self,
    ir::{Function, IR, Item},
};

/// Runs machine-independent intraprocedural optimization passes on the given
/// intermediate representation (_IR_), according to the specified `opts`.
pub fn optimize_ir(ir: &mut IR<'_>, opts: &Opts) {
    if !opts.any_passes_enabled() {
        return;
    }

    for item in &mut ir.program {
        if let Item::Func(func) = item {
            optimize_ir_func(func, opts);
        }
    }
}

/// Optimizes a single _IR_ function, applying the specified optimization
/// passes.
fn optimize_ir_func(func: &mut Function<'_>, opts: &Opts) {
    if func.instructions.is_empty() {
        return;
    }

    loop {
        if opts.fold {
            compiler::opt::passes::fold_ir_const(func);
        }

        let mut cfg = CFG::new(func);

        if opts.uce {
            compiler::opt::passes::unreachable_code(&mut cfg);
        }

        if opts.copy_prop {
            compiler::opt::passes::propagate_copy(&mut cfg);
        }

        if opts.dse {
            compiler::opt::passes::dead_store(&mut cfg);
        }

        if !cfg.apply(func) {
            break;
        }
    }
}
