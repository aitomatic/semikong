# Emerging Patterns from Knowledge Documentation

As we document more experiences, patterns emerge naturally. Here are the patterns that have surfaced from our CMP knowledge base:

## Pattern: The Tuesday Problem

**Observation from multiple engineers:**
- "Line went down Monday night after maintenance"
- "Tuesday morning, removal rate was 15% low"
- "Took us 4 hours to figure out the conditioner was misaligned"

**Pattern Recognition:**
After seeing this 8 times in 6 months, we realized:
1. Monday maintenance often involves pad changes
2. Conditioner alignment gets disturbed during pad change
3. Effects aren't visible until production restarts
4. Engineers assume it's a process issue, not mechanical

**Emerging Structure:**
```
Maintenance Event → Pad Change → Conditioner Check → Alignment Verification
```

See: [Post-Maintenance Checklist](post-maintenance-checklist.md)

## Pattern: The 3:00 PM Dip

**Multiple observations:**
- "Removal rate drops around 3 PM daily"
- "Uniformity gets worse too"
- "Always on hot days"

**Root Cause Discovery:**
The cleanroom HVAC system reduces cooling in the afternoon to save energy. CMP tools are sensitive to ambient temperature:
- Slurry temperature increases 2-3°C
- Pad expands slightly
- Removal rate increases (per 10°C rule)
- But uniformity degrades due to thermal gradients

**Solution:**
Adjusted HVAC schedule to maintain ±1°C during production hours.

## Pattern: The New Guy Effect

**Consistent observation:**
- "New operators get 20% more defects first month"
- "Not training - they know the procedure"
- "Seasoned operators can 'feel' when something's wrong"

**Pattern Analysis:**
Experienced operators develop tacit knowledge:
- Sound of motor indicates pad condition
- Vibration patterns reveal slurry flow issues
- Visual cues from wafer surface
- Timing intuition for endpoint detection

**Knowledge Capture:**
We're now documenting these "intangible" indicators:
- [Motor Sound Library](motor-sound-library.md)
- [Visual Defect Guide](visual-defect-guide.md)
- [Timing Intuition Training](intuition-training.md)

## Pattern: The Hidden Correlation

**Data mining revealed:**
- Defectivity spikes correlate with barometric pressure drops
- Not immediately obvious (24-48 hour lag)
- Affects all tools simultaneously

**Investigation:**
Pressure changes affect:
1. Slurry chemistry (gas solubility)
2. Pad porosity (micro-expansion)
3. Air bubble formation in lines

**Emerging Practice:**
Now we monitor weather forecasts and adjust:
- Increase slurry flow slightly before pressure drops
- Condition pads more frequently
- Add degassing cycles to slurry lines

## Pattern: The Expert Consensus

**Interesting phenomenon:**
When 3+ senior engineers independently mention the same "trick," it becomes part of the tacit knowledge base.

**Recent examples:**
1. "Always tap the conditioner twice before starting" (removes loose diamonds)
2. "Listen for the 'slurry song' - changes when flow drops"
3. "Check Tuesday's data first when investigating trends"

**Formalization Process:**
1. Document the individual observations
2. Verify with other engineers
3. Test scientifically if possible
4. Add to standard procedures if validated

See: [Expert Consensus Tracker](expert-consensus.md)

## Pattern: The Failure Cascade

**Observed sequence:**
1. Minor issue ignored (small RR drift)
2. Compensating adjustment made (increase pressure)
3. Secondary effect triggered (pad glazing accelerates)
4. Major failure occurs (uniformity loss + defects)
5. Emergency stop required

**Emerging Structure:**
```
Minor Drift → Compensation → Secondary Effect → Major Failure
     ↑              ↑              ↑              ↑
  Detection    Over-correction   Cascade      Emergency
```

**Prevention Strategy:**
- Address root cause, not symptoms
- Understand compensation side effects
- Monitor for cascade indicators
- Stop at first sign of cascade

See: [Failure Cascade Prevention](failure-cascade.md)

## From Patterns to Ontology

As these patterns solidify, they suggest ontological structure:

### Emerging Classes:
- `MaintenanceEvent` with `postMaintenanceChecks`
- `EnvironmentalCondition` with `compensationActions`
- `ExpertObservation` with `validationStatus`
- `FailureCascade` with `interventionPoints`

### Emerging Relationships:
- `maintenanceEvent` → `triggers` → `conditionerCheck`
- `environmentalChange` → `requires` → `compensation`
- `expertConsensus` → `validates` → `bestPractice`
- `minorDrift` → `cascadesTo` → `majorFailure`

### Emerging Rules:
- Post-maintenance checklist mandatory
- Environmental compensation within 2 hours
- Expert consensus requires 3+ confirmations
- Cascade intervention within 30 minutes

## The Knowledge → Structure Evolution

1. **Month 1-3**: Document everything (no structure)
2. **Month 4-6**: Notice recurring themes (loose patterns)
3. **Month 7-9**: Identify root causes (stronger patterns)
4. **Month 10-12**: Formalize best practices (emerging structure)
5. **Year 2+**: Codify into ontology (formal structure)

This is how knowledge-first ontology development works - structure emerges from understanding, not the other way around.

---

*See also: [From Tacit to Explicit](tacit-to-explicit.md)*
*Next: [Formalizing Patterns](formalizing-patterns.md)*

*Pattern recognition based on 18 months of knowledge capture*