# Design rules

1. **Module docs stay coherent with code on every commit.** If a commit
   changes a module's public surface, responsibility boundaries,
   schema, or layout assumptions, update `docs/modules/<module>.md` in
   the same commit.

2. **Keep [vision.md](vision.md) in mind and openly discussed.** Surface
   tradeoffs against the long-term vision when proposing non-trivial
   work; flag changes that constrain or contradict it.
