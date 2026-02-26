//! _MIR x86-64_ Optimization
//!
//! Optimization passes that operate on an _x86-64_ machine intermediate
//! representation (_MIR_) to improve code generation and efficiency.

pub mod coalesce;
pub mod liveness;
pub mod optimize;
pub mod register_alloc;

pub use coalesce::coalesce_loop;
pub use liveness::RegisterLiveness;
pub use optimize::optimize_x86_64_mir;
pub use register_alloc::{RegisterType, allocate_registers};
