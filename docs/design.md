# Design rules

Project-wide rules that must be followed during coding. Each rule is
load-bearing — read before opening a PR.

## Documentation

1. **Module docs must stay coherent on every commit.** Each module under
   `src/` has a corresponding doc at `docs/modules/<module>.md`. If a
   commit changes a module's public surface, responsibility boundaries,
   schema, failure modes, or path/layout assumptions, the matching doc
   must be updated in the same commit. CI / review will reject a commit
   that drifts the code from its doc. Trivial internal refactors that
   leave the public contract unchanged do not require a doc edit.

2. **Module docs are design notes, not tutorials.** Aim for ~1–2 pages
   rendered: purpose, responsibility boundaries (what the module owns
   vs explicitly does NOT own), key decisions table, failure-mode
   table, future work. No code listings beyond signatures. Use
   mermaid sparingly for architecture or data flow when it earns its
   keep.

3. **One doc per module.** When a module gains a major implementation
   (e.g. a second `DataSource`), it goes as a section in the existing
   module doc, not a new file. Split only if the module itself is
   split.

## Code

4. **No premature abstraction.** Don't introduce a generic when there's
   one concrete use. Three similar lines is better than a wrong
   abstraction. Grow abstractions only when a second concrete use
   forces them.

5. **`missing` vs `nothing`.** `missing` means "the data is absent";
   `nothing` means "the computation failed / skip this entry". Don't
   blur the two.

6. **`Float64` for domain values.** Prices, strikes, volumes, IVs.

7. **No comments that restate the code.** Only write a comment when
   the *why* is non-obvious — a hidden constraint, a workaround, an
   invariant a reader would not infer. Never add comments referencing
   the current task or PR.

8. **Validate at module boundaries, trust internally.** Constructors
   and protocol entry points may validate their inputs. Internal
   helpers do not re-check what their callers already guarantee.

## Tests

9. **Every module has tests under `test/<module>/`.** New public
   surface ships with tests in the same commit.

10. **Regression tests for every fixed bug.** When a bug is fixed, a
    test that fails on the old code and passes on the new must land
    with the fix.

## Commits

11. **Small, deliberate commits.** Each commit is one self-contained
    change with a rationale in the message. The branch's history
    should read as a story.

12. **Commit messages never mention Claude, AI, or assistant tooling.**
