# Contributing

## Setup

```bash
python -m pip install -e ".[plots]"
PYTHONPATH=src python -m unittest discover -s tests -v
```

`.[plots]` is what `make figures` needs. The tests run without it.

## Design rules

These are the properties the reported results depend on. A patch that breaks one is
rejected even when the tests pass, because the tests cover only part of the space.

1. No change under `src/capacity_tunnel/` may contact a host other than `127.0.0.1`. The channel is loopback by design.
2. Every number in a document must exist in `results/results.json`, and `make eval` must regenerate it.
3. The entropy-only baseline stays in the code. It is the comparison the claim rests on; deleting it would make the result unfalsifiable.
4. The estimator's assumptions (recurring alphabet, Markov-1 entropy rate) are stated in `docs/DESIGN.md`. A patch that widens them widens the document too.

## Changing a reported number

Every figure and table in `README.md` and `docs/EVALUATION.md` comes from
`results/results.json`. To change a number you re-run the evaluation, commit the
regenerated file, and let the diff show what moved. Hand-editing a table is the one
change that is always rejected; `scripts/check_repro.py` exists to catch it.

## Pull requests

Say what you measured, not only what you changed. If the patch moves a reported
value, include the before and after. If it changes performance, include the command
and the machine, since that is the only way the number means anything.

## Reporting a problem

Include the command, the input, and the output. For a wrong number, name the
`results.json` field you compared against.

Security issues go through [SECURITY.md](SECURITY.md), not a public issue.

## Style

- Python 3.11+, `from __future__ import annotations`, type hints on public functions.
- Line length 110 or less. CI runs flake8 with the F-class checks only.
- ASCII in source. Documents are English.
- No runtime dependency beyond what `pyproject.toml` already declares.
