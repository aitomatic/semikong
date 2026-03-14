# Prompt Template: 08-Materials Curation (Sammy)

Use this prompt to run a full content-curation pass for `ontology/08-materials/**`.

## Prompt

Run as **Sammy** (Semicont Ontologist) and execute a full curation pass for `ontology/08-materials/**`.

Mandatory startup:
1. Read `ontologist/AGENTS.md`
2. Read `ontologist/02-rules/{boundary-policy.yaml,curation-gates.yaml}`
3. Read `ontologist/01-docs/subontology-quality-checklist.md`
4. Read `ontologist/04-context/{TASKS.md,STATE.md,SESSIONS.md,WORKLOG.md}`
5. Run `python ontologist/05-analytics/tools/ontology_audit.py` before edits

Task scope:
- Curate `08-materials` module-by-module (including placeholder modules), with industry-general depth only (stop before company-specific differentiation).

Hard requirements for each edited module:
1. Use internet-accessible authoritative sources (not model-only synthesis).
2. Build a source packet (URLs + short rationale for authority).
3. Add/upgrade:
   - core classes
   - object properties
   - datatype properties
   - OWL axioms (disjointness/restriction/cardinality as appropriate)
   - SHACL validation shapes
   - unitized numeric modeling (not comments-only units)
   - external mapping stubs (conservative predicates)
4. Add provenance metadata with concrete source links:
   - `dc:source`, `dc:rights`, `prov:hadPrimarySource`, `prov:wasInformedBy`
5. Assign module maturity status (`scaffold` / `baseline` / `curated` / `validated`) and note remaining gaps.

Process discipline:
1. Update `SESSIONS.md` when starting/finishing.
2. Keep `TASKS.md`, `STATE.md`, `WORKLOG.md` current during the run.
3. Run `python ontologist/05-analytics/tools/ontology_audit.py` after edits and report results.
4. Do not commit unless explicitly requested.

Deliverables:
1. Updated `ontology/08-materials/**` TTL files.
2. A concise module-by-module completion table against the 10 checklist dimensions.
3. Source packet summary with links used.
4. Audit output summary and unresolved gaps.
