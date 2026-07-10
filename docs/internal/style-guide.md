# CosmicGrain Documentation Style Guide

This page defines reusable documentation patterns for CosmicGrain.

## Physics Page Template

Every dust physics page should use this structure:

1. At a Glance
2. Overview
3. Scientific Motivation
4. Physical Model
5. Numerical Implementation
6. Algorithm
7. Parameters
8. Source Files
9. Validation
10. Related Processes
11. Why This Matters
12. References

---

## At a Glance

```markdown
!!! abstract "At a Glance"

    **Purpose:** Brief statement of what this module does.

    **Inputs:** Main physical or numerical inputs.

    **Outputs:** Main simulation outputs.

    **Runtime Cost:** Low / Moderate / High.

    **Physics Ladder:** S0, S1, ..., S10.

    **Status:** Implemented / experimental / planned.
```

---

## Scientific Motivation

```markdown
!!! info "Scientific Motivation"

    Explain why this process matters astrophysically.
```

---

## Numerical Implementation

```markdown
!!! example "Numerical Implementation"

    Explain what CosmicGrain actually does in the code.
```

---

## Algorithm

```markdown
!!! example "Algorithm"

    1. Compute stellar ejecta.
    2. Apply dust condensation efficiency.
    3. Search neighboring gas.
    4. Create PartType6 dust particles.
    5. Initialize dust properties.
```

---

## Source Files

```markdown
!!! quote "Relevant Source Files"

    | File | Purpose |
    |------|---------|
    | `src/...` | Description |
```

---

## Validation

```markdown
!!! success "Validation"

    - Mass conservation checked.
    - Restart compatibility checked.
    - Expected behavior verified in the physics ladder.
```

---

## Why This Matters

```markdown
!!! tip "Why This Matters"

    Connect the implementation back to the science.
```
