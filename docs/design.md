# Design rules

1. **Module docs stay coherent with code on every commit.** If a commit
   changes a module's public surface, responsibility boundaries,
   schema, or layout assumptions, update `docs/modules/<module>.md` in
   the same commit.

2. **Keep [vision.md](vision.md) in mind and openly discussed.** Surface
   tradeoffs against the long-term vision when proposing non-trivial
   work; flag changes that constrain or contradict it.

3. **Surface rule changes; don't absorb them.** When a prompt implies
   a new rule or a change to an existing one (in this file or any
   module doc), propose the edit explicitly before applying it.

4. **Keep [status.md](status.md) current.** When work materially
   advances progress toward [vision.md](vision.md), changes the
   current sequence of work, or moves an item between active and
   future scope, update status.md in the same commit.

5. **Check current community conventions before locking design
   decisions.** When introducing a new module, type hierarchy, or
   public API shape, look up current Julia (or relevant ecosystem)
   conventions for naming, layout, interface declaration, and
   testing structure. Cite the findings in the design discussion so
   the choice is traceable.
   Record the findings in a *Conventions consulted* section of the
   module's `docs/modules/<module>.md`, one entry per decision naming
   the source checked. That section is the durable record; a design
   discussion may point to it but is not where the citation lives.

6. **Module docs hold commitments, not lookups.** A module doc explains
   boundaries, invariants, data flow, and the decisions this project
   committed to -- which of several defensible arrangements it picked,
   what it would break to change, and what must stay true across
   changes. The code already says what it does; only a doc can say what
   we agreed to.

   The test: **would someone with the code in front of them answer this
   correctly on their own?** A signature, a parameter list, a return
   type, the steps of an algorithm, an enumerated API, "how do I read a
   surface" -- all yes, so all lookups. They regenerate on demand, they
   go stale the moment the code moves, and they are what makes one
   module doc restate its neighbour. Delete them; a docstring carries
   what survives (rule 8). A reason, an alternative rejected, an
   invariant that holds across the module -- all no, so all commitments,
   and they belong here.

   Where a concrete detail is load-bearing, state the rule and let the
   instance be an example inside it, never a section of its own: an
   invariant about *any* derived provider, not a tour of one. Prefer
   invariants and boundaries over specifics that drift -- magic numbers,
   defaults, internal data structures, incidental library names.

7. **Empty means temporal absence only.** An empty result from any read
   means "served, and nothing at this instant". Every other unanswerable
   question has a name: nothing serves this selector, two records with
   two answers, a derivation that keeps failing past its bound, a leg
   that cannot honestly be priced. Name it with an error type or a
   counted field, never with an ordinary empty result. A consumer that
   cannot tell "not yet" from "not ever" correctly concludes it has
   nothing to do, and the run completes with no positions and no
   diagnostic.

8. **Comments and docstrings do not restate the module doc.** A
   **docstring** states the contract of the thing it is attached to:
   what it computes or returns, and what it refuses, by name. One
   function or type, nothing wider. A **comment** explains the code it
   sits on -- a local complexity, or a decision in *that* code that
   would otherwise look wrong and invite a "fix". A **file header**
   names what the file holds in a line or two; it does not narrate the
   module's architecture, invariants or conventions, which are
   `docs/modules/<module>.md`.

   The test: if a passage would still be true and useful with the
   surrounding code deleted, it belongs in the doc. Where a comment
   must point at a doc, name the module rather than a path -- paths go
   stale silently when a module is renamed.

9. **Folders are stages of a run, not financial objects.** The top level
   of `src/` names what a stage *does* -- price, serve data, decide,
   simulate, record, evaluate, orchestrate, persist, show -- with one
   `docs/modules/<folder>.md` per folder. A market object therefore
   appears in as many folders as it has aspects: a surface's math in
   `pricing`, its record and kind contract in `data/kinds`, the provider
   that serves it in `data/providers`, and the same three for a curve.
   Collecting every aspect of one object into a folder of its own would
   be a principle only if every object were cut that way; one such
   folder standing beside eight stages is a special case, and the layout
   should not carry one.

   The seam that keeps this honest: `pricing` may name record types --
   `build_surface` consumes a chain -- but nothing in it may reach the
   protocol, a provider, a cut or an experiment. A dependency that wants
   to run the other way says a stage boundary is in the wrong place, not
   that the rule needs an exception.

10. **Never write future work into a module doc.** Forward-looking work
    lives in one place, [status.md](status.md)'s backlog, where an item
    is concrete parked work with a decision behind it. A module doc says
    what the module *is* and what it committed to; a roadmap inside one
    ages into a promise nobody made -- a reader cannot tell parked work
    from abandoned work, and two docs drift into contradicting each
    other, and the backlog, about what is coming.

    A doc may name a deferral only when the backlog carries it, and then
    in a clause rather than a section: enough to say the gap is
    deliberate and not an oversight, with the plan left in the backlog.
    Reversibility is not future work -- what it would take to change a
    decision is part of stating it (rule 6) and stays.

11. **Repo-wide style, recorded once.** No `get_` prefix on accessors; a
    bang only on mutation; files `include`d into the one top-level
    module with no submodules. Pkg.jl's package guide fixes only
    `src/<Pkg>.jl` and leaves the rest to logical grouping, so the
    layout is the project's own choice and rule 9 is where it is made.

    A module doc's *Conventions consulted* is for that module's own
    naming decisions (rule 5) and never for these: a convention that
    holds everywhere belongs here, where it is stated once, rather than
    in whichever module doc was written first.

