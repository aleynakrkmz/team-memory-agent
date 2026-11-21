## Reading of the results (by hand)

*Written by WS3 (Gözde Nur Özdemir) after run 2, 21 Nov 2026.*

**Team decision (WS3 proposal; to be confirmed by the team):** the suggestion is accepted as it is.

| Role | Model | Family | Why |
|---|---|---|---|
| Agent | `qwen3:8b` | Qwen (Alibaba) | The only model that solved the agent case (right task and owner within 6 steps). Extraction F1 0.621 with no guessed owners; the ~20k-token B2 prompt was read in full in 45 s. `gemma3:12b` has a higher F1 (0.724) but Ollama reports no tool support for it, so it cannot run the agent loop. |
| Generator A | `mistral-nemo:12b` | Mistral (Mistral AI) | The highest first-try pass rate (3 of 5 meetings) and 4 of 5 after three tries, 55 s per meeting. `qwen3:8b` passed all 5 meetings but cannot be a generator, because it is the agent's model. |
| Generator B | `gemma3:12b` | Gemma (Google) | 4 of 5 meetings after three tries, 2 of 5 on the first try. `llama3.1:8b` also passed 4 of 5 but only 1 on the first try; `granite3.3:8b` passed 3 of 5. |
| Checker (datagen check iv) | `llama3.1:8b` | Llama (Meta) | The best extraction F1 (0.448) among the models that are neither the agent nor a generator. Measured as a checker after run 2 (see below). |

Generators A and B come from different families, and neither comes from the agent's family.

**Manual review of the check (iv) flags.** Run 2 flagged 8 changes (7 in meetings written by `qwen3:8b`, 1 by `mistral-nemo:12b`). All 8 were read in the saved meetings, and all 8 are true flags: the generator added a commitment that the scenario does not have, for example "I'll make sure the endpoints are documented by Tuesday", "I'll also add a feedback form to the login" and "could we maybe add a little loading spinner?". No false flag was found, so the run 1 problem is fixed. Missed changes (false negatives) were not measured.

**Is `llama3.1:8b` good enough as the checker?** In run 2, check (iv) was done by `qwen3:8b`, so the suggested checker was measured separately: `--measure-checker llama3.1:8b` ran check (iv) with it on the same 50 saved meetings (no new meetings were written).

- Of the 4 meetings with a true flag, `llama3.1:8b` flagged 3 and missed 1.
- It flagged 5 meetings that `qwen3:8b` had not flagged. Read by hand: 3 are false (a status remark read as an assignment, "payment gateway integration is lagging"; a due date that is not set; an event label "E14" left in the text read as a task name) and 2 are borderline (a side remark that a query "should be done by end of day"; a request for "the data schema by Friday" with no owner).

A false flag only costs one more try, so it does not harm the data. A missed flag is the risk: an extra change would then be missing from the answer key. We still keep `llama3.1:8b`, because the alternative, `qwen3:8b`, is the agent's model: if the agent's own model decided which test meetings are clean, extra changes that this model cannot see would stay in the test data unnoticed, and the agent would never be penalised for missing them. To cover the missed flags, WS3 reads a sample of the generated test meetings by hand before the freeze. The team decides.

**Other observations**

- Most remaining failures are generator mistakes that checks (i) and (ii) caught: a planned event without its label, or an owner or date that the line does not state. A failed meeting is written again (at most three tries).
- `gemma3:12b` sometimes puts the event label in the middle of a line instead of at the end (for example "By the 1st. [E7] Someone needs to own this."). The label check caught all 5 cases. The parser could also accept a label in the middle of a line.
- With both A and B, one meeting in five failed all three tries. The datagen pipeline therefore needs a clear rule for such a meeting.
- `mistral-nemo:12b` sometimes starts lines with Markdown asterisks ("Ayşe: ** Alright"). The transcript parser should remove them.

**Limits.** One development project of 5 meetings per model and one run. Generation uses temperature 0.8, so pass rates will vary between runs; they are enough to choose the models, not to report data quality.
