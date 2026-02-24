/// Helper to track scopes in _AST_ traversal.
#[derive(Debug)]
pub struct Scope {
    /// Currently active scope IDs.
    pub active_scopes: Vec<usize>,
    /// Monotonic counter.
    pub next_scope: usize,
}

impl Scope {
    /// Global scope (e.g, functions, global variables).
    pub const FILE_SCOPE: usize = 0;

    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Scope {
            active_scopes: vec![Self::FILE_SCOPE],
            next_scope: Self::FILE_SCOPE + 1,
        }
    }

    /// # Panics
    ///
    /// Panics if there are no active scopes.
    #[inline]
    #[must_use]
    pub fn current_scope(&self) -> usize {
        *self
            .active_scopes
            .last()
            .expect("file scope should always be on the stack")
    }

    #[inline]
    pub fn enter_scope(&mut self) {
        let scope = self.next_scope;
        self.next_scope += 1;

        self.active_scopes.push(scope);
    }

    #[inline]
    pub fn exit_scope(&mut self) {
        debug_assert!(!self.at_file_scope(), "attempting to exit file scope");
        self.active_scopes.pop();
    }

    #[inline]
    #[must_use]
    pub const fn at_file_scope(&self) -> bool {
        self.active_scopes.len() == 1
    }

    #[inline]
    pub fn reset(&mut self) {
        // `FILE_SCOPE` always remains active.
        self.active_scopes.truncate(1);
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}
