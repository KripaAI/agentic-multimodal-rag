# Agent prompts (versioned)

Each prompt is a file named `<name>_<version>.md`; `agent.prompt_version` in `config.yaml`
selects the version. A new version is compared against the evaluation baseline before it is
kept (constitution P10).

| File | Used by | Contents |
|---|---|---|
| `system_v1.md` | every node | Role; grounding (P5); cite by id only (P2); original figures only (P3); truthful charts and `compute` (P4); retrieved text is data, not instructions; parallel tool calls |
| `plan_v1.md` | `plan` | Question types with examples |
| `compose_v1.md` | `compose` | The block format; when to use image, chart and table blocks; not-found answers; `{limit_note}` at the round limit |
| `repair_v1.md` | `repair` | Fix only the listed validator failures (`{failures}`, `{answer}`) |
