## What this changes

<!-- One paragraph. If it moves a reported number, give the before and after. -->

## Kind

- [ ] Defect fix
- [ ] New protocol, parser or detector
- [ ] New measurement or experiment
- [ ] Documentation
- [ ] Refactor with no behaviour change

## Design rules

- [ ] No change under `src/capacity_tunnel/` may contact a host other than `127.0.0.1`. The channel is loopback by design.
- [ ] Every number in a document must exist in `results/results.json`, and `make eval` must regenerate it.
- [ ] The entropy-only baseline stays in the code. It is the comparison the claim rests on; deleting it would make the result unfalsifiable.
- [ ] The estimator's assumptions (recurring alphabet, Markov-1 entropy rate) are stated in `docs/DESIGN.md`. A patch that widens them widens the document too.

- [ ] `PYTHONPATH=src python -m unittest discover -s tests -v` passes
- [ ] Reported numbers regenerated from `results/results.json`, not hand-edited
- [ ] Documentation updated where behaviour is user-visible

## Evidence

<!-- For a new detector or measurement: the input that triggers it, and one that must
not. False positives matter as much as detections. -->

## Notes for review

<!-- Ordering assumptions, overlap policy, performance trade-offs. -->
