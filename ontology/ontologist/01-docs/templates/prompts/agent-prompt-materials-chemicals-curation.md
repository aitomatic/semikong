# Agent Prompt: Materials Chemicals Curation

Use this prompt to run a focused ontology curation task for `08-materials/01-chemicals`.

## Prompt

Run as the Semicont ontologist using the control plane in `ontologist/`.

Start here:
1. `ontologist/AGENTS.md` (primary runbook)
2. `ontologist/02-rules/{boundary-policy.yaml,curation-gates.yaml}` (enforced rules)
3. `ontologist/01-docs/subontology-quality-checklist.md` (required design dimensions)
4. `ontologist/03-skills/` and `ontologist/01-docs/workflows/` (execution playbooks)
5. `ontologist/05-analytics/tools/ontology_audit.py` (required audit tool)
6. `ontologist/04-context/{TASKS.md,STATE.md,WORKLOG.md,SESSIONS.md}` (state tracking)

Task:
Curate `ontology/08-materials/01-chemicals/ontology.ttl` as real production ontology content (not placeholder cleanup).

Content goals:
1. Define industry-general scope and boundary (stop before company-specific differentiation).
2. Add concrete classes for process chemicals and specs/properties.
3. Add key object relationships and datatype properties.
4. Add measurable value modeling with units/quantity pattern.
5. Add minimum OWL axioms (disjointness + at least one restriction/cardinality).
6. Add SHACL-in-TTL shapes for required fields and numeric sanity checks.
7. Add provenance/copyright-safe source documentation per rules.

Competency questions the module must support:
1. Which chemicals are suitable for a process step?
2. Which chemicals are incompatible with a material or process condition?
3. Which chemicals satisfy purity/concentration requirements?
4. What storage/hazard constraints apply?

Process requirements:
1. Follow AGENTS start checklist.
2. Update `TASKS.md`, `STATE.md`, `WORKLOG.md`, `SESSIONS.md` during execution.
3. Run `python ontologist/05-analytics/tools/ontology_audit.py` at the end.
4. Report what was curated, what remains, and audit output.
5. Do not commit unless explicitly requested.
