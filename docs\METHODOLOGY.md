# Methodology and reporting boundary

## Paired design

For a fixed model and seed, all methods use the same persisted client partition and the same base
adapter initialization. Local minibatch seeds are derived only from `(seed, round, client_id)`.
Changing the defense therefore does not silently change the examples or their order.

## Effective contribution coefficients

The paper defines HRCW and HSSR over each method's final nonnegative effective contribution
coefficient. FedSApro records both the raw aggregation coefficient and the final effective
coefficient before normalization:

- FedAvg uses sample-count weights.
- FedProx and SCAFFOLD use sample-count weights multiplied by the submitted update norm relative
  to the round's median standard-benign update norm. This makes local shrinkage/drift correction
  visible without treating either optimizer as an explicit malicious-client detector.
- IndexGuard and FLAME use zero for rejected clients and the sample-count weight for retained
  clients.
- FLTrust uses the rectified cosine trust score after norm matching to the server root update.

This convention is explicit because FedProx/SCAFFOLD do not natively output per-client trust
weights, while the paper reports non-zero suppression rates for both.

## Attacks

Label flipping reverses the binary label on exactly 30% of each malicious client's persisted
examples. Textual backdoor poisoning prefixes `I watched this 3D movie.` to exactly 10% of each
malicious client's examples and sets the label to positive. Test accuracy is always measured on
clean, trigger-free sets, matching Table I.

## Reproduction status

Paper targets are immutable reference values in `configs/paper/targets.yaml`. Measured outputs are
written only by the runner. The validator never replaces measurements with targets. Candidate
runs and rejected seeds remain distinguishable from accepted single-seed reproductions.

