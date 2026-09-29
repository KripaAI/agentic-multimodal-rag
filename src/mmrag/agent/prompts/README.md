# Agent prompts (versioned)

Each prompt is a file named `<name>_<version>.md`; `agent.prompt_version` in `config.yaml`
selects the version. A new version is compared against the evaluation baseline before it is
kept (constitution P10). Written in the implementation step, after the skeleton is reviewed.

| File | Used by | Contents |
|---|---|---|
| `system_v1.md` | every node | Role; grounding (answer only from retrieved evidence, P5); cite by id only (P2); original figures only (P3); chart rules (P4); memory is never evidence (P13); say "not found in the documents" when the corpus doesn't answer |
| `plan_v1.md` | `plan` | Question types with examples; round limits; which tools suit which type |
| `compose_v1.md` | `compose` | The block format (spec §5.4); when to use image, chart and table blocks; at the round limit, state what is missing |
| `repair_v1.md` | `repair` | Fix only the listed validator failures; change nothing else |
