# Vision

The long-term shape of this repo, kept short and revisited often.

## Research

A research codebase where option strategies can be backtested end-to-end.
Each backtesting experiment is defined declaratively in a config (inputs,
strategy, schedule, evaluation), is rerunnable at will, and writes its
results into a knowledge base that accumulates over time -- so prior runs
remain queryable and comparable rather than evaporating into ad-hoc
notebooks.

## Trading

Eventually a separate automatic-trading codebase, sharing the strategy and
modelling layers with research, plus a thin broker-interaction layer that
turns strategy decisions into live orders. The split keeps research free
of broker concerns and trading free of experiment scaffolding.

## What this means in practice

- Every component should be usable from both a research config and a live
  loop. Avoid coupling logic to either context.
- Reproducibility is a first-class concern: experiments are
  config-defined, results are persisted, and the persistence layer is
  designed so prior conclusions stay accessible.
- Components added today should not block the eventual trading split --
  prefer interfaces that work for both.
