# Prompt Template: 07-WFE-06 Etch Systems Curation (Sammy)

Use this prompt to run a focused, source-grounded curation pass for `ontology/07-wfe/06-etch-systems/ontology.ttl`.

## Prompt

Run as **Sammy** (Semicont Ontologist) and curate `ontology/07-wfe/06-etch-systems/ontology.ttl`.

Mandatory startup:
1. Read `ontologist/AGENTS.md`
2. Read `ontologist/02-rules/{boundary-policy.yaml,curation-gates.yaml}`
3. Read `ontologist/01-docs/subontology-quality-checklist.md`
4. Read `ontologist/01-docs/templates/prompts/facets-dimensions-template.md`
5. Read `ontologist/04-context/{TASKS.md,STATE.md,SESSIONS.md,WORKLOG.md}`
6. Run `python ontologist/05-analytics/tools/ontology_audit.py` before edits

Hard source policy:
1. Use ONLY Internet-accessible primary sources (official standards bodies, equipment/process references, public technical documentation).
2. Do NOT rely on model-synthesized claims without source backing.
3. Respect copyright/fair use:
- Paraphrase; do not copy long text verbatim.
- Keep only minimal identifiers/titles needed for references.

Etch-systems depth target:
1. Expand hierarchy beyond top-level etch categories to practical sub- and sub-subcategory depth.
2. Keep leaves industry-general and reusable; avoid company-specific internals.
3. Cover major etch modalities and chamber/process distinctions where broadly accepted.

Required facets/dimensions:
Implement and report against all facets in `facets-dimensions-template.md`, including:
1. Scope/boundary
2. Taxonomy depth
3. Entities/roles
4. Process/workflow semantics
5. Capabilities/constraints
6. Settings/parameterization
7. Specifications/quality
8. Measurement/units
9. Lifecycle/state
10. SHACL validation
11. Provenance/rights
12. External alignment

Process discipline:
1. Update `TASKS.md`, `STATE.md`, `WORKLOG.md`, `SESSIONS.md` during the run.
2. Run `python ontologist/05-analytics/tools/ontology_audit.py` after edits.
3. Do not commit unless explicitly requested.

Deliverables:
1. Updated `ontology/07-wfe/06-etch-systems/ontology.ttl`.
2. Categorized findings report:
- boundary
- legacy path
- TTL quality
3. Prioritized fix summary with rationale.
4. Completion table across all required facets/dimensions (`done`/`partial`/`missing`).
5. Source packet with URLs used and short authority rationale.
6. Proposed edits (no commit) to:
- `ontologist/04-context/TASKS.md`
- `ontologist/04-context/STATE.md`
- `ontologist/04-context/WORKLOG.md`
