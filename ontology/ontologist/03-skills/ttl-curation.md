# Skill: TTL Curation

## Goal

Add or refine ontology terms in Turtle while preserving layer taxonomy and naming conventions.

## Procedure

1. Identify target layer/subcategory path in `ontology/`.
2. Edit only relevant `ontology.ttl` file(s).
3. Keep IRIs stable and human-readable.
4. Reuse shared semantics from `00-shared` when cross-layer.
5. Run `python ontologist/05-analytics/tools/ontology_audit.py`.
6. Update runtime files in `ontologist/04-context/`.

## Done Criteria

- No boundary violations.
- No stale or legacy path references introduced.
- Runtime tracking updated.
