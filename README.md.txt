# Bias Amplification in Elite Competition: A Simulation Study

## Research Question

How does a small, persistent bias get amplified into significant inequality
through repeated competitive selection?

This project models a stylized elite competition system to isolate and 
demonstrate one specific mechanism: **bias as a linear amplifier**. The 
question is not whether inequality exists, but how a fixed perceptual bias 
propagates through institutional selection into cumulative ability gaps —
even when the two groups start from identical distributions.

---

## Model

The simulation runs a discrete-time competition between two groups (A and B)
over T rounds. The core design prioritizes causal clarity over realism.

**Population.** 2N individuals, N per group. Initial abilities are drawn from
the same distribution — N(50, 10) — with no built-in advantage for either
group.

**Bias mechanism.** In each round, a perceptual bias β is subtracted from
Group B's *perceived* ability before selection. True abilities are never
directly altered by bias. This isolates the mechanism: bias operates
entirely at the evaluation layer.

**Selection.** A fixed proportion top_k of the population is selected each
round, with selection probabilities determined by a softmax over perceived
abilities, controlled by temperature parameter τ. Lower τ sharpens
competition (high-ability individuals win more deterministically); higher τ
flattens it (more randomness).

**Accumulation.** Winners receive an ability increment δ (plus noise).
There is no upper bound and no elimination. This models a *positional
competition* over opportunities, not survival.

The causal chain is:

    bias → selection probability → opportunity allocation → ability growth
         → feeds back into next round's competition

---

## Key Findings

**1. Bias produces stable, linear amplification.**  
Under baseline parameters, the ability gap between A and B grows linearly
over time. No threshold or tipping-point dynamics are observed under the
baseline specification — the amplification is continuous and proportional.
This is the core result.

**2. β determines the amplification rate.**  
Linear regression on gap(t) time series across M=100 simulations confirms
that the slope of gap growth increases with β (mean R² = 0.98). Bias
intensity does not just shift the final gap — it sets the *speed* at which
inequality accumulates.

**3. Opportunity capture stabilizes quickly.**  
Group A's winner ratio rises sharply in early rounds and then stabilizes,
while outcome inequality continues to accumulate. The mechanism does not
require accelerating exclusion — a stable distributional advantage in
opportunity access is sufficient to generate growing outcome inequality.

**4. Institutional parameters have separable primary effects.**  
A joint sensitivity scan over top_k and τ shows that the two parameters
affect different dimensions of the amplification process. top_k (competition
intensity) strongly influences the overall magnitude of accumulated
inequality, while τ (selection sharpness) modulates how strongly bias
translates into differential selection probabilities. Both parameters
interact, but their primary effects operate through distinct channels.

---

## Project Structure
```

bias-amplification/
│
├── stochastic-injustice-sim.py   # All simulation and analysis code
│   ├── Module 1                  # Population initialization
│   ├── Module 2                  # Single-round update (Softmax selection)
│   ├── Module 3                  # Full simulation loop
│   ├── Module 4                  # Monte Carlo repetition & parameter scans
│   └── Module 5                  # Visualization functions
│
└── README.md

```
To reproduce all results, run the main execution block at the bottom of
`stochastic-injustice-sim.py`. All analyses run sequentially with printed
progress output.

**Dependencies:** numpy, pandas, scipy, matplotlib, seaborn

---

## Parameters

| Parameter | Description                               | Baseline |
|-----------|-------------------------------------------|----------|
| N         | Group size (each group)                   | 200      |
| T         | Number of rounds                          | 100      |
| β (beta)  | Perceptual bias against Group B           | 5        |
| top_k     | Fraction selected per round               | 0.2      |
| δ (delta) | Ability increment per win                 | 1.0      |
| ε_std     | Noise on ability increment                | 0.5      |
| τ (tau)   | Softmax temperature (selection sharpness) | 1.0      |
| M         | Monte Carlo repetitions                   | 100      |

---

## Limitations and Scope

This model is intentionally minimal. Several features of real competitive
systems are excluded by design.

**No elimination or resource constraint.** Abilities grow without bound and
no one exits the competition. This means the model captures positional
dynamics — who gets ahead — rather than survival or market-clearing
dynamics. The absence of a capacity constraint is a scope decision, not an
oversight.

**No feedback from outcomes to bias.** In reality, growing inequality may
reinforce or reduce bias over time (through legitimation effects or
backlash). This model holds β fixed to isolate the mechanical amplification
process.

**No within-group heterogeneity in bias exposure.** All B-group members
receive the same β penalty. Heterogeneous or intersectional bias structures
are outside the current scope.

These exclusions make the core mechanism legible. The claim is not that this
model is a complete account of inequality — it is that the mechanism it
isolates is real, tractable, and worth understanding on its own terms.
