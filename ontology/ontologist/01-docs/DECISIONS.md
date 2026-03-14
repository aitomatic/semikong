# Decisions

## 2026-03-10: Internet-Sourced Evidence And Copyright Guardrails

- Decision: Require ontology curation to use both model reasoning and external internet references (prefer primary/authoritative sources), with provenance notes recorded in `ontologist/04-context/WORKLOG.md` for substantive edits.
- Why: Reduces hallucination risk and improves traceability/reviewability of semantic decisions.
- Decision: Enforce copyright and IP safeguards in curation rules: avoid large verbatim copying, use paraphrased semantic extraction, allow only short fair-use quotations when necessary, and prohibit disallowed proprietary/confidential source material.
- Why: Preserves open-source safety and legal hygiene while keeping ontology development evidence-based.

## 2026-03-10: Enhanced Provenance Requirements

### Decision
All ontology TTL files must include comprehensive provenance documentation using W3C PROV-O and Dublin Core vocabularies to ensure copyright/IP compliance and source traceability.

### Rationale
- Meet evidence triangulation requirements from boundary policy
- Provide in-file documentation of fair-use justifications
- Enable automated validation of source attribution
- Align with open science best practices

### Changes Made
1. Updated `02-rules/curation-gates.yaml`:
   - Added `ttl_provenance_completeness` check
   - Added `class_level_provenance` check
   - Enhanced `evidence_traceability` to require TTL documentation

2. Updated `02-rules/boundary-policy.yaml`:
   - Enhanced `source_traceability` to require both TTL and context documentation
   - Updated `copyright_and_ip_compliance` to require dc:rights documentation

3. Updated `ontologist/AGENTS.md`:
   - Added provenance documentation requirements to execution checklist
   - Specified required vocabularies (PROV-O, DC)
   - Added fair-use documentation requirements

4. Created new artifacts:
   - `03-skills/provenance-documentation.md` - Practical guide for adding provenance
   - `01-docs/workflows/provenance-workflow.md` - Step-by-step workflow

### Implementation
Going forward, all new ontology modules must include:
- Module-level: dc:source, dc:rights, prov:hadPrimarySource
- Class-level (for substantive additions): prov:wasInformedBy, dc:source
- Fair-use justification in dc:rights where applicable

### Backlog
Existing TTL files created before this decision need retroactive provenance documentation. This will be addressed in a separate maintenance task.

## 2026-03-08: Three-View Organization

- Decision: Organize `semicont` as industry layers (`who/where`), semantic core (`what`), and profiles (`how applied`).
- Why: Balances adoption clarity and semantic rigor.

## 2026-03-08: Layer + Cross-Cutting Matrix

- Decision: Keep both value-chain layers and cross-cutting perspectives.
- Why: Layers support stakeholder onboarding; cross-cutting perspectives prevent duplicated semantics.

## 2026-03-08: Public/Private Boundary

- Decision: Keep public ontology assets open in `semicont`; keep implementation-specific automation in private environments.
- Why: Supports ecosystem adoption while preserving normal competitive IP boundaries.

## 2026-03-08: Canonical Design and Workflow Docs

- Decision: Add deterministic operating docs (`docs/design-spec.md`, `docs/source-policy.md`, `docs/methodology.md`, `docs/readiness-test.md`) as required references for agent execution.
- Why: Convert high-level direction into repeatable, testable execution.

## 2026-03-08: Canonical Serialization

- Decision: Use Turtle (`.ttl`) as canonical ontology serialization for initial implementation.
- Why: Human-readable, diff-friendly, and practical for iterative curation workflows.

## 2026-03-08: Agentic Export Pattern

- Decision: Keep Turtle as definitive source, create agentic exports in markdown/yaml for human consumption and tool integration.
- Why: Maintains semantic rigor while supporting the agentic documentation patterns proven in `~/.dana/ontologies`.

## 2026-03-08: Source Tier Policy

- Decision: Enforce Tier A/B/C source policy with explicit provenance requirements and quarantine flow for unclear rights.
- Why: Reduce copyright/licensing risk while preserving research throughput.

## 2026-03-08: Numbered Layer-First Ontology Layout

- Decision: Organize ontology directories by industry layer first, with numeric prefixes (`00-` through `09-`).
- Why: Matches audience navigation, keeps structure deterministic for agents, and preserves a stable shared area in `00-shared`.

## 2026-03-08: First Layer-06 Use Case Seed

- Decision: Seed layer 06 with `final-test-execution` as the first executable use-case module and keep Turtle as source of truth with colocated `-rules.yaml` and `-skills.md`.
- Why: Establishes a concrete template for packaging/final-test equipment workflows while preserving the `what` vs `how` separation.

## 2026-03-08: Layer-06 Foundation + Use-Case Split

- Decision: Introduce a dedicated layer-06 foundation module and refactor `final-test-execution` to hold only use-case-specific extensions.
- Why: Keeps foundational `what` concepts reusable across multiple layer-06 use cases while preserving focused use-case modules.

## 2026-03-08: Layer-06 Domain Split And Use-Case Directories

- Decision: Place layer-06 root ontology at `ontology/06-osat-packaging-test/ontology.ttl`, split domains under `osat/`, `packaging/`, and `test/`, and place each use case in its own directory (for example `test/final-test-execution/`).
- Why: Keeps layer-06 scoped to OSAT, packaging, and test semantics, avoids a flat file layout, and keeps each use case self-contained.

## 2026-03-09: Ontology Boundary Is TTL-Only

- Decision: `ontology/` remains source-of-truth TTL plus `ontology/README.md` only; rules, skills, workflows, and tooling artifacts are maintained under `ontologist/`.
- Why: Prevents format drift in ontology source and keeps operational artifacts clearly separated from semantic truth.
- Supersedes: prior practice of colocating 02-rules/skills artifacts with ontology modules.

## 2026-03-09: Ontologist Control-Plane Structure

- Decision: Organize `ontologist/` into `runtime/`, `governance/`, `rules/`, `skills/`, `workflows/`, `tools/`, `templates/`, and `notes/`.
- Why: Gives AI agents a deterministic navigation and execution model with clear separation of concerns.

## 2026-03-09: Agentic Map Reference Pattern

- Decision: Use `examples/cognitive-ontology/ks-4` as a reference pattern for agentic organization of 02-rules/03-skills/context/learning assets.
- Why: Reuses a proven structure while keeping this repository's ontology boundary contract intact.
