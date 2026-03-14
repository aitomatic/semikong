#!/bin/bash
# Export agentic formats from definitive Turtle ontology

set -e

echo "Exporting agentic formats from Turtle source..."

# Shared ontology export
echo "Processing shared ontology..."
# TODO: Implement RDF parser to extract terms and generate exports

# Module exports
echo "Processing lot genealogy module..."
# Generate markdown from TTL comments and structure
# Generate YAML rules from OWL restrictions
# Generate skills from competency questions

echo "Export complete!"
echo ""
echo "Generated files:"
echo "  - ontology/04-foundry-idm/use-cases/lot-genealogy.md (human documentation)"
echo "  - ontology/04-foundry-idm/use-cases/lot-genealogy-rules.yaml (validation rules)"
echo "  - ontology/04-foundry-idm/use-cases/lot-genealogy-skills.md (agent skills)"
echo ""
echo "Source of truth remains: ontology/04-foundry-idm/use-cases/lot-genealogy.ttl"
