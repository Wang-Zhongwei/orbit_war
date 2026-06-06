## Limitations
- didn't consider comets as target because they are not important
- 1v1 and 1v3 are indistinguishable in features
- only one cross attention layer
- no feature engineering for fleets aka no tick_to_arrive because other good player's must be good at aiming, policy network just need to learn an approximate sense of threat is good
- we don't know where the fleet is heading towards  

┌────────┬───────────────────────────────────┐
│ cutoff │ % of planet-bound fleets captured │
├────────┼───────────────────────────────────┤
│ T≤50   │ 95.46%                            │
├────────┼───────────────────────────────────┤
│ T≤75   │ 98.18%                            │
├────────┼───────────────────────────────────┤
│ T≤100  │ 99.86%                            │
├────────┼───────────────────────────────────┤
│ T≤150  │ 100.00%                           │
└────────┴───────────────────────────────────┘

┌───────┬───────┐
│  pct  │ ticks │
├───────┼───────┤
│ p50   │ 6     │
├───────┼───────┤
│ p75   │ 11    │
├───────┼───────┤
│ p90   │ 27    │
├───────┼───────┤
│ p95   │ 49    │
├───────┼───────┤
│ p99   │ 85    │
├───────┼───────┤
│ p99.9 │ 104   │
├───────┼───────┤
│ max   │ 106   │