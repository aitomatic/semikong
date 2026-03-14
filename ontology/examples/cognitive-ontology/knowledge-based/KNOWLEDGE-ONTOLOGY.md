# Knowledge-First Ontology: CMP Process Engineering

This is a knowledge-first ontology where understanding emerges from documented experience rather than predefined structure.

## How This Works

1. **Start with what practitioners know** - Document real experiences, problems, solutions
2. **Let patterns emerge** - Structure appears as knowledge accumulates
3. **Keep it conversational** - Written for humans first, machines second
4. **Link liberally** - Connections form organically through cross-references

## Core Knowledge Areas

### The CMP Learning Curve

When I first started in CMP, nobody told me that the pad conditioning rate is actually more critical than the slurry flow rate for uniformity. It took me three months of troubleshooting wafer-to-wafer variation to figure that out.

**Key insight**: Pad conditioning creates micro-scratches that maintain slurry transport. Without proper conditioning, the pad glazes and you get:
- Lower removal rate
- Poor uniformity
- Defectivity spikes

See: [Pad Conditioning Deep Dive](pad-conditioning-guide.md)

### Preston's Equation in Practice

The textbook says: Removal Rate = Kp × Pressure × Velocity

Reality says:
- Kp changes with pad life
- Pressure distribution isn't uniform across the wafer
- Velocity has radial and tangential components
- Temperature affects everything

**Real-world adjustment**: We maintain a lookup table of effective Kp values by:
- Pad hours (0-50, 50-100, 100-150, 150+)
- Slurry type (oxide, copper, tungsten)
- Platen combination

### The 3:00 AM Epiphanies

These are the things you learn at 3:00 AM when the line is down:

1. **Always check the conditioner diamond size first** - 80% of "pad problems" are actually conditioning issues
2. **Motor current trending tells you more than removal rate** - It's leading, not lagging
3. **Thermal cameras are worth their weight in gold** - Hot spots predict failures
4. **Never trust a single endpoint trace** - Always correlate multiple signals

See: [Emergency Troubleshooting Playbook](emergency-troubleshooting.md)

## Process Control Philosophy

### Why We Control What We Control

**Removal Rate**: Everyone focuses here because it's easy to measure, but it's actually a lagging indicator. By the time RR drifts, you've already made bad parts.

**Uniformity**: More important than RR but harder to measure. Within-wafer uniformity affects yield. Wafer-to-wafer uniformity affects customer confidence.

**Defectivity**: The silent killer. You can have perfect RR and uniformity but still ship bad parts if you're not watching defects.

See: [What to Watch When](process-monitoring-priorities.md)

## Equipment-Specific Wisdom

### Applied Mirra Secrets

The manual won't tell you that:
- Head retention force decreases by ~2% per month (springs fatigue)
- The optical endpoint detector drifts with temperature (±5°C = ±2% signal)
- Slurry flow meters are accurate to ±5% at best
- Pad life is 30% shorter on platen 1 due to higher usage

See: [Mirra Maintenance Insights](mirra-insights.md)

### Ebara F-REX Observations

What we learned after 6 months:
- Linear pad velocity is more stable than rotary
- But pad replacement takes 3x longer
- Endpoint detection is more sensitive but also more noisy
- Overall uptime is 5% better despite longer maintenance

## Decision Trees

### When Removal Rate Drops Suddenly

1. Check pad life first (<20 hours remaining? → Replace pad)
2. Check conditioner (Last conditioned >2 hours ago? → Condition now)
3. Check slurry flow (Readings stable? → Verify with graduated cylinder)
4. Check pressure (Calibrated this week? → Check calibration)
5. Check temperature (±2°C of setpoint? → Check heat exchanger)

See: [Full Decision Tree Visual](removal-rate-decision-tree.md)

### When Uniformity Degrades

1. Radial pattern? → Check pad conditioning profile
2. Azimuthal pattern? → Check carrier/platen speeds
3. Edge-fast? → Check retaining ring condition
4. Center-fast? → Check pad hardness/age

## Rules of Thumb

These aren't in textbooks but save hours of troubleshooting:

- **The 10°C rule**: Every 10°C increases removal rate by ~20%
- **The 100-hour rule**: Pads stabilize after 100 hours, then degrade after 300
- **The 3-sigma rule**: If uniformity degrades by >3σ, check conditioner first
- **The Tuesday rule**: Most problems appear on Tuesday (after Monday maintenance)

## Integration Points

### With Metrology
CMP doesn't exist in isolation. Key handoffs:
- Pre-CMP thickness sets the target
- Post-CMP uniformity drives next process window
- Defect detection affects rework decisions

See: [CMP-Metrology Interface](metrology-integration.md)

### With Diffusion
Thermal processes after CMP are affected by:
- Residual slurry chemistry
- Surface roughness
- Metallic contamination levels

See: [Downstream Process Impact](downstream-impact.md)

## Evolution of Understanding

This ontology grows as we learn. Recent additions:
- 2026-02: Added thermal cycling effects on pad life
- 2026-01: Updated Kp values for new slurry formulation
- 2025-12: Added motor current trending analysis

## How to Contribute

Found something new? Document it here:
1. Write what happened
2. Include context (equipment, conditions, time)
3. Explain what you learned
4. Link to related knowledge

See: [Contribution Guide](contributing.md)

---

*This is living knowledge. Last updated: 2026-03-08 by process engineer with 15 years CMP experience*