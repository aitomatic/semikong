# Benchmark Set

This directory stores external ontology snapshots used for calibration.

## Current Benchmarks

1. `SemicONTO-0.2.ttl`
- Source: https://github.com/huanyu-li/SemicONTO/blob/main/ontology/0.2/SemicONTO.ttl
- Raw: https://raw.githubusercontent.com/huanyu-li/SemicONTO/main/ontology/0.2/SemicONTO.ttl
- Curator/creator (metadata): Huanyu Li (`dcterms:creator`)
- Focus: semiconductor materials, experiment workflows, and FAIR data interoperability.

2. `DigitalReference.ttl`
- Source: https://github.com/tibonto/dr
- Raw: https://raw.githubusercontent.com/tibonto/dr/master/DigitalReference.ttl
- Curator/creator (metadata): George Dimitrakopoulos (`terms:creator`)
- Focus: semiconductor supply chain + production/lifecycle reference model.

3. `IOF-Core.rdf`
- Source: https://github.com/iofoundry/ontology
- Raw: https://raw.githubusercontent.com/iofoundry/ontology/master/core/Core.rdf
- Curator/creator (metadata): IOF Core Working Group (`dcterms:creator`)
- Focus: cross-industry manufacturing foundational ontology terms.

4. `SAREF4INMA-v2.1.1.ttl`
- Source: https://saref.etsi.org/saref4inma/
- Raw: https://saref.etsi.org/saref4inma/v2.1.1/saref4inma.ttl
- Curator/creator (metadata): Raul Garcia-Castro and ArcelorMittal Global R&D Asturias (`dcterms:creator`)
- Focus: smart industry/manufacturing interoperability extension in the SAREF family.

## Usage

- Run calibration:
  - `python ontologist/05-analytics/tools/benchmark_calibration.py`
- Use results to update `ontologist/04-context/STATE.md` benchmark tables.
