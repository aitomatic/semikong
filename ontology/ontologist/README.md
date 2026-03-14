# Ontologist Directory

`ontologist/` is the operational control plane for ontology curation.

## Purpose

- Provide agents and humans with one place for process, policy, workflow, and status tracking.
- Keep operational artifacts out of `ontology/` so ontology source remains clean and TTL-first.

## Boundary Contract

- `ontology/` contains canonical semantics in Turtle (`.ttl`) and one guide file: `ontology/README.md`.
- `ontologist/` contains rules, skills, workflows, tools, and curation state.

## Layout

- `00-index.yaml`: navigation index for agents/tools.
- `01-docs/`: decisions, templates, workflows.
- `02-rules/`: YAML gate and boundary policies.
- `03-skills/`: concise how-to playbooks for curation tasks.
- `04-context/`: live state (`STATE`), queue (`TASKS`), claims (`SESSIONS`), and log (`WORKLOG`).
- `05-analytics/`: local scripts to audit quality and detect drift.
- `06-learning/`: reusable lessons and retrospectives.
- `AGENTS.md`: complete agent runbook and starting point.

## Benchmarking

- External benchmark snapshots live in `05-analytics/benchmarks/`.
- Current benchmark set includes:
  - SemicONTO 0.2
  - Digital Reference
  - IOF Core
  - SAREF4INMA
- Run calibration with:
  - `python ontologist/05-analytics/tools/benchmark_calibration.py`
- Run ontology gates with:
  - `python ontologist/05-analytics/tools/ontology_audit.py`
- Record benchmark deltas in `04-context/STATE.md` status tables.

## Sub-Ontology Quality Assessment

The subontology-quality-checklist.md defines 10 dimensions for assessing module maturity:

1. Scope Boundary
2. Core Classes
3. Object Properties
4. Datatype Properties
5. Logical Axioms And Constraints
6. Units And Quantities
7. External Mappings
8. Provenance And Evidence
9. Validation
10. Maturity Status

Run quality measurement:
```bash
python ontologist/05-analytics/tools/measure_subontology_status.py [subdirectory]
```

Recent assessments:

- `07-wfe/06-etch-systems`: 1 curated, 7 scaffold modules (see `04-context/07-wfe-06-etch-systems-quality-assessment.md`)

## Quick Start

1. Open `AGENTS.md`.
2. Follow the "Minimal Start Checklist".
3. Execute one task through the full loop and update `04-context/`.

## Design Note

This structure follows the spirit of `examples/cognitive-ontology/ks-4` with renumbered directories (`01-...` onward) for predictable, agent-friendly navigation.
