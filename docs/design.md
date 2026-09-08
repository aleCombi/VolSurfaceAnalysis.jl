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

6. **Keep module docs lean and architectural.** Module docs should
   explain boundaries, invariants, data flow, and consequential design
   decisions. Include usage notes only when they are crucial to using
   the module correctly; expect routine examples and API walkthroughs
   to move later into dedicated API docs or examples. Favor invariants
   and boundaries over implementation specifics (magic numbers, internal
   data structures, incidental library names) that drift as code
   changes.

7. **Empty means temporal absence only.** An empty result from any read
   means "served, and nothing at this instant". Every other unanswerable
   question has a name: nothing serves this selector, two records with
   two answers, a derivation that keeps failing past its bound, a leg
   that cannot honestly be priced. Name it with an error type or a
   counted field, never with an ordinary empty result. A consumer that
   cannot tell "not yet" from "not ever" correctly concludes it has
   nothing to do, and the run completes with no positions and no
   diagnostic.
