# Rule transplant: moving a learned rule between models by editing weights

Model A is fine-tuned on problems built from 100 new operators (generated with [Opulse](https://github.com/mkj69/opulse)), giving A′. An agent gets the weights of A, A′ and B and must edit B so that B learns the operators while keeping B's original abilities.

**Site:** https://mkj69.github.io/rule-transplant/

- `index.html` — summary: main table, setup, each setting, findings, what can and cannot be claimed, limitations, verbatim prompts.
- `trajectories/guided.html` — full trajectories of the guided Fable runs (agent given the list of new symbols).
- `trajectories/unguided.html` — full trajectories of the unguided Fable runs (100 operators, same architecture).
- `trajectories/runs/` — one Markdown trajectory per run, including the 1,000- and 10,000-operator runs and the cross-architecture runs.

Not included: operator definitions, random seeds, training data, held-out sets and model weights.

Status: work in progress (2026-11-09). All guided runs are finished. The definitions-only arm has one run, stopped at operator 19.
