//! Register Allocation (_x86-64_)
//!
//! Handles the assignment of virtual registers to physical registers on
//! the _x86-64_ architecture. This pass ensures efficient use of CPU registers,
//! resolves conflicts, and inserts spills and reloads as needed during code
//! generation.
