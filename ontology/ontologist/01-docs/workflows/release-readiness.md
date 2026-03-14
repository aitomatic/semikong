# Workflow: Release Readiness

1. Run `python ontologist/05-analytics/tools/ontology_audit.py`.
2. Confirm no non-TTL files under `ontology/` except `ontology/README.md`.
3. Confirm no legacy paths remain in ontology or ontologist docs.
4. Confirm context state reflects current completed/blocked work.
5. Summarize risks and unresolved decisions.
