# Source and Copyright Policy

## Purpose

Define what sources agents may use and how provenance and copyright risk are controlled.

## Source Tiers

1. Tier A (Allowed by default)
- Your own authored content
- Public standards/specifications with clear reuse terms
- Public-domain or permissive-license technical references

2. Tier B (Allowed with review)
- Vendor/public documentation with limited reuse language
- Academic papers with unclear redistribution rights

3. Tier C (Disallowed for ingestion)
- Paywalled or access-restricted proprietary content without permission
- Unlicensed scraped corpora
- Content with explicit no-reuse/no-derivative restrictions for this use

## Agent Rules

1. Never ingest Tier C sources.
2. If license/rights are unclear, mark as `needs_review` and stop ingestion.
3. Prefer paraphrase over verbatim copying.
4. Keep quotations minimal and necessary for attribution.
5. Do not treat generated/synthetic text as factual authority.

## Required Provenance Fields

Every term/assertion derived from external material must include:

1. `source_url`
2. `source_title`
3. `source_license`
4. `retrieved_at` (ISO date)
5. `citation_type` (`quote`, `paraphrase`, `synthetic`, `expert-input`)
6. `confidence` (`low`, `medium`, `high`)
7. `review_status` (`draft`, `reviewed`, `approved`, `rejected`)

## Synthetic Content Policy

1. Synthetic content is allowed for tests/bootstrapping.
2. It must be tagged `citation_type=synthetic`.
3. It cannot be the sole evidence for stable ontology terms.
4. Stable terms require at least one non-synthetic source.

## Escalation Policy

When uncertain:

1. Quarantine candidate terms/assertions.
2. Record `needs_review` with rationale.
3. Continue only after explicit approval.
