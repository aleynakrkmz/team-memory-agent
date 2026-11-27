# Evaluation contract and its status

The evaluation contract was fixed before any result, in the system design document of **15 November 2026**
(Section 6.5, with the metrics of 6.2-6.4 and the budget of 6.6). This file restates it so the repository holds it,
and tracks what has been run against it. The contract is not changed here: where the work departs from it, the
departure is listed with its reason (rubric: "report every comparison the contract planned, in the form it
planned").

## The seven fields (from the design document)

| # | Field | Contract |
|---|---|---|
| 1 | Task | One item is one meeting, processed in order within its project. Output: the proposed task changes and the resulting task list. Main condition: transcripts. Secondary: Kokoro audio of the held-out synthetic projects through faster-whisper. Unit of analysis: the project. |
| 2 | Baseline | B1: fixed rules with BM25 matching, no LLM; rules fixed before test data are generated. B2: one prompt with all transcripts so far and the current task list; its prompt developed on the development set with the same effort as the agent's. Same model, settings, validator and approval path for all LLM setups. Ablations: A1 agent loop off, A2 search off, A3 clarification off. |
| 3 | Development vs. held-out | Development: 6-7 projects from model A. Held out: 2-3 projects from model B (about 10-15 meetings) with unseen task and person names, 3-5 hand-written meetings, 3-5 labelled AMI meetings. Key checked and frozen; then rules, prompts, limits and thresholds frozen with a git tag before the first test run. Only the WS3 lead reads held-out material, and she does not change prompts, rules or limits. |
| 4 | Scoring rule | Change-level F1 (micro, per type) in both modes; accumulated task-state and workload accuracy in carry-forward mode; unsupported-update rate with a 30-proposal hand check; clarification with simulated answers. Results per meeting and per project, then mean and range across projects. Synthetic, hand-written and AMI results reported separately. |
| 5 | Repeated runs | Three live runs per project for the agent, B2 and each ablation; one run for B1 (deterministic). Fresh database per run. Ablations in reset-to-gold mode only. Mean and range per project. |
| 6 | Resource budget | Per meeting: at most 25 calls, 40,000 tokens, 5 minutes (B2: 2 calls, 30,000 tokens, 5 minutes). Whole test run: at most 30 hours of compute. No money cost (local Ollama). Anything not measured is reported as "not measured", never as zero. |
| 7 | Failures and timeouts | Every failure stays in the denominator. A meeting that fails completely counts all its gold changes as missed; in a partly processed meeting, unprocessed or stopped cases are missed. Wrong proposals are false positives. Every result states its execution mode and the model name and version. |

## Status on the development set (dev set v1, 27 November 2026)

| Planned | Status | Where |
|---|---|---|
| Development set: 6-7 projects from model A | Done: 7 projects x 5 meetings, mistral-nemo:12b, checked by llama3.1:8b; manifest with SHA-256 | `datagen/generate.py`, `docs/data_generation.md` |
| Answer-key check before freezing | Automatic checks (i)-(iv) done; manual read of all 35 meetings found about 5 doubtful key items (4%); the owner hint is added to the generator prompt for the next sets | `docs/data_generation.md` |
| Change F1, both modes | Carry-forward: done for agent, B1, B2. **Reset-to-gold: not yet run** | `eval/harness.py`, `docs/evaluation.md` |
| Task-state accuracy (carry-forward) | Computed in every carry-forward run | `scores.json` |
| Workload accuracy | **Not measured yet** | - |
| Unsupported-update rate | Automatic proxy in every run; **30-proposal hand check prepared, not done** | `eval/unsupported_check.py` |
| Clarification measures (asking precision and recall, resolution rate, questions per meeting) | Questions per meeting recorded; **the other three not computed yet** | - |
| Three runs for agent and B2, one for B1 | Done in carry-forward mode: agent F1 0.592 (0.579-0.599), B1 0.607, B2 0.412 (0.408-0.421). **Reset-to-gold not repeated** | `docs/evaluation.md` |
| Mean and range across projects | Micro F1 over all projects reported; **per-project mean and range to add** | `eval/compare.py` |
| Ablations A1-A3 (reset-to-gold, 3 runs) | **Not in the harness yet** | - |
| BM25 vs. nomic-embed-text (development only) | **Not run** | - |
| Resource budget | Real Ollama token counts and model time per meeting are recorded for agent and B2 | `compare.py` columns |
| Failures in the denominator, execution mode on every result | Done | `eval/harness.py`, `run_record.json` |
| Denied-approval test (20% rejections, development only) | **Not run** | - |
| Prompt injection and Rule 2 | Done on an injected copy of the development set (28 attacks, 14 legitimate lines); not in the contract's seven fields, part of the safety analysis | `eval/injection.py` |

## Departures so far

- **B2's prompt** was written on the two sample projects, not developed on the development set with the agent's
  effort (field 2). It must be tuned on the development set by someone who has not read the held-out material
  before the freeze, or the report states that it was not.
- **B1's rules** were rewritten on the development set (version 2) after version 1, written on the sample projects,
  scored F1 0.00-0.14 there. Both versions are reported; the rules are frozen before test data are generated, as
  field 2 requires.

## Before the first test run

1. Run what is marked "not yet" above on the development set, or record why it is dropped.
2. Generate the held-out set (2-3 projects with gemma3:12b and the test name and task lists), check and freeze its
   key; add the hand-written and AMI meetings.
3. Freeze rules, prompts, limits and thresholds with a git tag; from then on nothing in `prompts/`, `config/`,
   `eval/b1.py` or `app/baseline/` changes.
4. Run every planned comparison on the held-out set and report all of them, whichever way they come out.
