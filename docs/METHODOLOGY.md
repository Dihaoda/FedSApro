# Methodology and reporting boundary

## Paired design

For a fixed model and seed, all methods use the same persisted client partition and the same base
adapter initialization. Local minibatch seeds are derived only from `(seed, round, client_id)`.
Changing the defense therefore does not silently change the examples or their order.

The efficient profile applies a server learning rate of 2.0 to every aggregation method. This
uniformly scales the aggregated update and is recorded in each resolved configuration; it is a
round-compression strategy for the two-round single-GPU study, not a change to any defense score.
The frozen paper profile keeps the conventional server learning rate of 1.0.

## Classification objective

FedSApro uses Qwen's two-class sequence-classification head and applies rank-8 LoRA only to the
query and value projections. The small `score` head is trainable and checkpointed with the
adapter; the transformer backbone remains frozen. The two rows of the score head are initialized
from Qwen's pretrained language-model output vectors for the configured sentiment-label tokens.
Inputs end in the documented classification prompt, so the initial two-logit classifier is the
efficient equivalent of next-token label scoring without materializing full-vocabulary logits.

## Effective contribution coefficients

The paper defines HRCW and HSSR over each method's final nonnegative effective contribution
coefficient. FedSApro records both the raw aggregation coefficient and the final effective
coefficient before normalization:

- FedAvg uses sample-count weights.
- FedProx and SCAFFOLD use sample-count weights multiplied by the submitted update norm relative
  to the round's median standard-benign update norm. This makes local shrinkage/drift correction
  visible without treating either optimizer as an explicit malicious-client detector.
- IndexGuard uses zero for rejected clients and the sample-count weight for retained clients.
- FLAME uses zero for rejected clients; a retained client's diagnostic coefficient is its
  sample-count weight multiplied by its norm-clipping scale.
- FLTrust's diagnostic coefficient is the rectified cosine trust score multiplied by the
  root-to-client norm-matching scale.

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

