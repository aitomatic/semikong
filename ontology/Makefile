PYTHON ?= python

.PHONY: help audit benchmark export-agentic

help:
	@echo "Available targets:"
	@echo "  audit          Run ontology boundary and legacy audit"
	@echo "  benchmark      Run benchmark calibration against included references"
	@echo "  export-agentic Run the current agentic export script"

audit:
	$(PYTHON) ontologist/05-analytics/tools/ontology_audit.py

benchmark:
	$(PYTHON) ontologist/05-analytics/tools/benchmark_calibration.py

export-agentic:
	bash scripts/export-agentic.sh
