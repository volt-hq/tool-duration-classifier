# Tool duration classifier

A standalone experiment for [Volt](https://github.com/volt-hq/Volt), tracked with
the [Volt Roadmap](https://github.com/orgs/volt-hq/projects/1).

A trained, dependency-free logistic regression model that estimates whether a tool
call will take **more than two seconds**. It runs locally in Node. It does not execute
the supplied commands, select async execution, assess dependency safety, or modify
Volt's tool scheduling.

The included model is a prototype trained on **79 hand-authored synthetic examples**.
Its labels express expectations for installed tools in a moderate repository; they
are not measured execution times. Actual duration depends on machine, repository,
caches, network, arguments, and failures. Do not interpret its score as a calibrated
probability or use these results as evidence of production accuracy.

## Try it

From the repository root, with Node 22.19 or newer:

```sh
node cli.mjs predict --command "npm run test"
node cli.mjs predict --command "git status --short"
node cli.mjs evaluate
node cli.mjs benchmark
node --test model.test.mjs
```

No install step is needed. `npm run check` runs syntax checks and all ten tests.
GitHub Actions runs checks on Linux and Windows with Node 22 and 24.

The first two examples return `long` and `short`, respectively. No dependencies,
GPU, model service, or network access are required. Predictions never launch the
command being classified. `--help` describes the CLI.

For other tools, put a call in a JSON file and use `predict --input call.json`:

```json
{ "tool": "read", "arguments": { "path": "package.json" } }
```

Output includes `durationClass`, `longScore`, `thresholdMs`, and training `basis`.
Scores at or below 0.25 return `short`; scores at or above 0.75 return `long`;
intermediate scores and unseen tool names return `uncertain`. These are fixed
experimental cutoffs, not calibrated confidence guarantees. Unknown commands for a
known tool can still produce an incorrect confident prediction.

## Initial results

The holdout has 36 examples across 17 manually separated command/task groups, with
18 short and 18 long labels. Related variants remain together; shared concepts such
as `test` and `build` still occur in both splits. This is a small synthetic sanity
check, not a representative workload benchmark. No parameters were tuned against
the holdout after evaluation.

| Predictor | Correct / decided | Coverage |
| --- | --- | --- |
| Learned model, forced decision at 0.5 | 31 / 36 (86.1%) | 100% |
| Learned model, with uncertainty band | 15 / 16 (93.8%) | 44.4% |
| Keyword rules | 34 / 36 (94.4%) | 100% |
| Always short | 18 / 36 (50.0%) | 100% |

The model currently loses to the rules baseline. It confuses some print-only
commands and writing a build script with actually executing slow work. With the
uncertainty band it identifies only 6 of 18 long calls as long; the other 12 abstain.
It also confidently misclassifies one short print command. `evaluate` emits the
confusion counts, long precision/recall, coverage, and every forced-decision error.

The model artifact is 10,073 bytes. One Windows x64 run on Node v24.20.0 measured
about **0.0031 ms mean / 0.0047 ms p95** for warm inference, including feature
extraction, over 10,000 predictions of seed-sized inputs. File read, JSON parse,
and model validation took about 1.46 ms, excluding Node startup. Timings vary by
machine; rerun `benchmark` locally. The CLI starts Node for each invocation; a
harness experiment should load once and call `predict` in process.

## Training and measured data

Recreate the included model deterministically:

```sh
node cli.mjs train
```

`seed-data.json` contains the dataset. Each example has `group`, `split` (`train`
or `test`), `tool`, `arguments`, and either a synthetic `label` (`short` or `long`)
or an actual nonnegative `durationMs`. Never supply both. A measured example is:

```json
{
  "group": "repo-a-npm-unit-tests",
  "split": "train",
  "tool": "bash",
  "arguments": { "command": "npm run test:unit" },
  "durationMs": 18452
}
```

Wrap examples in `{ "thresholdMs": 2000, "examples": [...] }`. Both splits must
contain both duration classes. A duration equal to the threshold is short. Cancelled,
timed-out, or otherwise incomplete observations should be excluded: their elapsed
time is not a completed execution duration. Synthetic labels must be reconsidered
if the duration threshold changes; measured labels are derived automatically.

```sh
node cli.mjs train --data local-measurements.json --model local-model.json
node cli.mjs evaluate --data local-measurements.json --model local-model.json
node cli.mjs predict --model local-model.json --command "npm run test"
```

Keep local measurements outside the repository if they contain private paths or
command arguments. There is no automatic collection, execution, or upload. Group
repeated commands, wrappers, and variants together; for real evaluation, hold out
entire repositories or sessions where possible. The validator rejects groups and
identical feature vectors crossing splits; semantic near-duplicates still require
manual review. Training never uses test labels or adjusts thresholds on the test set.

Next useful evidence: collect actual completed tool durations, retrain, and evaluate
on unseen repositories. Compare against keyword rules and per-command duration
history. A wait-briefly-then-yield baseline needs an execution/scheduling experiment;
this text-only dataset cannot establish latency or token savings from that policy.

## Model details

The feature extractor uses tool identity, argument word tokens, and adjacent token
pairs. It lowercases and caps serialized argument text at 4,096 characters, applies
signed FNV-1a hashing into 2,048 slots, then L2-normalizes the sparse vector. Arguments
after the cap are ignored. It is not a shell parser and does not understand quoting,
shell aliases, scripts, file sizes, or dependencies. Large argument serialization
can cost more than the seed-input benchmark suggests.

Training uses logistic loss with L2 regularization and 500 fixed full-batch gradient
steps. The artifact stores float32-rounded weights as JSON numbers, plus a bias and
training metadata. In-memory JS arrays and Node use more memory than the artifact.
No hardcoded command classifications participate in training or inference; the
keyword rules are a separate evaluation baseline.

The method follows standard [feature hashing](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-hashing)
and [linear logistic classification](https://scikit-learn.org/stable/modules/sgd.html).
The implementation uses Node built-ins and does not depend on scikit-learn.

## When to try it in Volt

The prototype can begin observation-only evaluation now. Live scheduling needs a
separate harness integration; this repository does not yet provide one.

1. Run predictions in shadow mode while keeping existing scheduling. Record actual
   completed execution durations, model predictions, repository/session groups, and
   inference overhead. Capture job completion time, not an async launch response.
2. Retrain on measured durations and evaluate on held-out repositories or sessions.
   Freeze thresholds and parameters before the final evaluation. Compare precision,
   recall, uncertainty coverage, and runtime overhead with keyword rules and command
   duration history. Choose acceptable error rates for the intended workload first.
3. Compare the model, keyword rules, existing scheduling, and a short wait before
   yielding on repeatable end-to-end tasks. Keep the main model and harness settings
   fixed, account for warm/cold caches, repeat runs, and report variability. Measure
   task completion time, model tokens/turns, and successful task completion.
4. Use the predictor in scheduling only if it improves those end-to-end measures
   without reducing task correctness. Limit the experiment to background-eligible
   work; the harness still owns dependencies, cancellation, and completion delivery.

There is no defensible fixed sample count or accuracy percentage yet. The decision
depends on workload coverage, error costs, and repeatable scheduling gains. The
synthetic benchmark currently favors simple rules, so it does not justify enabling
the classifier by default.
