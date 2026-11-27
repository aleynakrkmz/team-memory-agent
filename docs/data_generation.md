# Generating the development and test projects

`datagen/generate.py` builds the synthetic projects used for evaluation (design §12.1, §12.2, Figure 2).
The sample projects in `samples/` were written from fixed templates and are too easy: B1 reaches F1 0.95 on
them and its rules were written while reading them. The development set is written by a model instead.

## Commands

```bash
# development set (7 projects x 5 meetings)
uv run python datagen/generate.py --split dev --projects 7 --generator mistral-nemo:12b --checker llama3.1:8b

# test set: generator B, test names and tasks; only the WS3 lead reads it
uv run python datagen/generate.py --split test --projects 3 --generator gemma3:12b --checker llama3.1:8b \
    --out datagen/out/test

# template meetings, no model (to try the code)
uv run python datagen/generate.py --split dev --projects 2 --backend mock --out /tmp/dev_mock
```

## Steps

| Step | Who | What |
|---|---|---|
| A. Team and scenario | code, fixed seed | `common.generate_project`: team, tasks, every task change of every meeting (the answer key). Dev and test use separate name and task lists. |
| B. Meetings | generator model | `common.render`: the model gets the participants and the agenda and writes a messy meeting. Every line that states an agenda item ends with a label such as `[E3]`. Labels are removed from the transcript and kept in the answer key. |
| C. Automatic check | code + checker model | `common.auto_check`: (i) every event has a labelled line, (ii) owners named and dates right, (iii) an owner left open on purpose is really not named, no label left in the speech, (iv) a different model lists the changes it finds; a change the scenario does not have is flagged. |
| Retry | code | A failed meeting is written again with a new seed, at most 3 times. If a meeting fails 3 times the project is dropped (later meetings depend on it) and the next seed is used. |
| D. Manual check | a person | `attempts/` keeps every attempt with its labels and problems. |
| E. Freeze | code + git | `manifest.json`: SHA-256 of every file, models and Ollama digests, seeds, code commit, pass rates. `manifest.sha256`: one hash over the manifest. |

Seeds: `dev01` uses seed 1001, `dev02` 1002, and so on; test projects start at 9001. A dropped seed is listed in
`dropped_seeds`, so the set can be rebuilt with the same models.

The script refuses to write into a folder that is not empty, so a frozen set is never overwritten.

## Output

```
datagen/out/dev/
  manifest.json, manifest.sha256
  generation_log.csv          one row per attempt: project, seed, meeting, attempt, passed, reasons, seconds
  attempts/<project>_seed<n>/m01_try1.txt ...
  dev01/agent_inputs/dev01/   team.json, meetings.json, meeting_01.transcript.txt ...
  dev01/answer_keys/dev01/    scenario.json, meeting_01.evidence.json ...
```

Each project folder works directly with the harness:

```bash
uv run python eval/harness.py datagen/out/dev/dev01 --config b1 --mode carry
```

## Rules

- Development projects may be read by everyone and used to change prompts, rules and limits.
- Test projects are read only by the WS3 lead. Nobody changes a prompt, rule or limit after seeing them.
- The generator of the test set (gemma3:12b) is not the agent model and not the development generator.
