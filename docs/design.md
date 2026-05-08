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
   advances the rebuild or changes its order, update status.md in
   the same commit.
