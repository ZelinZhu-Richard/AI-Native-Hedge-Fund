# Project Meridian -- TODO Tracker
# Last Updated: March 16, 2026
# READ THIS FILE AT THE START OF EVERY CODING SESSION.
 
---
 
## FUND THESIS
Markets systematically misprice regime transitions. We combine LLM agents (detecting regime shifts from unstructured data) with a Mixture-of-Experts forecasting engine (adapting predictions across market regimes) to capture alpha during exactly the moments when other systematic funds lose money.
 
---
 
## FINAL ARCHITECTURE (post iron-audit)
 
```
┌─────────────────────────────────────────────────────┐
│ LAYER 1: Data + Features + Regimes                  │
│ 3 LLM agents (filing, earnings, macro)              │
│ + PCA + 50+ features + regime detection             │
│ + Devil's advocate (different LLM)                  │
└──────────────────────┬──────────────────────────────┘
                       │ structured features + regime signals
                       ▼
┌─────────────────────────────────────────────────────┐
│ LAYER 2: MoE Signal (3 experts + gating)            │
│ TFT (stable) + MAML (transition) + N-BEATS (trend)  │
│ Gating network routes by regime                      │
└──────────────────────┬──────────────────────────────┘
                       │ return forecasts + confidence
                       ▼
┌─────────────────────────────────────────────────────┐
│ LAYER 3: Strategy + Execution                       │
│ Pairs trading + directional + Black-Litterman        │
│ Disagreement scoring -> position sizing              │
└──────────────────────┬──────────────────────────────┘
                       │ proposed positions
                       ▼
┌─────────────────────────────────────────────────────┐
│ LAYER 4: Risk + Circuit Breakers                    │
│ VaR/CVaR + hardcoded limits + vol scaling            │
│ Correlation monitor + sector exposure caps           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
╔═════════════════════════════════════════════════════╗
║ CIRCUIT BREAKERS -- Hardcoded. No override. Ever.  ║
╚══════════════════════╤══════════════════════════════╝
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ LAYER 5: Quantitative Monitoring                    │
│ Rolling Sharpe per expert + feature drift            │
│ Overfitting detection + gating audit (ALL MATH)     │
└─────────────────────────────────────────────────────┘
```
 
### Agent Architecture (Multi-LLM)
```
Filing Analyst (Claude) ────┐
                            ├── Disagreement Score ── Position Sizing
Earnings Analyst (GPT-4) ──┤
                            │
Macro Analyst (DeepSeek) ──┤
                            │
Devil's Advocate (rotates ──┘
  whichever model the
  primary analyst DIDN'T use)
```
 
### Key Architecture Decisions (from iron audit)
- 3 MoE experts, NOT 4. Add 4th only if backtesting shows a specific gap.
- Black-Litterman portfolio optimization, NOT quantum-inspired. Classical and proven.
- Quantitative monitoring (math), NOT LLM-based monitoring (vibes).
- Parallel multi-LLM agents with disagreement scoring, NOT multi-round debate.
- Devil's advocate runs on a DIFFERENT LLM than the primary analyst.
- Domain-expert agent names with explicit frameworks, NOT investor persona names.
- Disagreement between agents automatically reduces position sizing.
 
---
 
## CURRENT SPRINT: Sprint 1 (Foundation + Core Alpha Engine)
## Timeline: Mar 15 - Apr 15, 2026
 
### Completed
- [x] Day 1: Project scaffolding, core module (exceptions, logging, types), data providers (Yahoo Finance), DuckDB storage with upsert, data validation (gaps, outliers, splits, staleness), ingestion orchestrator with resume capability
- [x] Day 2: Feature engineering (50+ features across technical, volatility, cross-sectional, macro), feature registry, feature store (long format DuckDB, wide format retrieval), anti-lookahead test suite
- [x] Day 3: Rolling PCA, HMM + K-Means regime clustering, regime detector, regime analysis + visualization, Docker support, TODO.md, ARCHITECTURE.md update
 
### Completed (cont.)
- [x] Day 4: Backtesting framework
  - Walk-forward validation engine (504d train, 63d test, 63d step)
  - Portfolio tracker (positions, cash, equity curve)
  - Transaction cost models (realistic Almgren-Chriss, zero, pessimistic 2x)
  - Performance metrics (Sharpe, Sortino, drawdown, Calmar, VaR/CVaR, etc.)
  - 5 benchmark strategies (buy-hold, momentum 12-1, mean reversion, SPY, 60/40)
  - Anti-lookahead validators (signal timing, data snooping, turnover, window consistency, return plausibility)
  - Backtest report generation (text + strategy comparison tables + plots)
  - SignalGenerator protocol (interface for all future models)
  - Integration tests (full pipeline: data -> signal -> backtest -> validate -> report)
 
### Upcoming (Sprint 1)
- [ ] Day 5: TFT Expert implementation
  - Implement Temporal Fusion Transformer
  - Multi-horizon forecasting (1d, 5d, 21d returns)
  - Interpretable attention outputs
  - Implement SignalGenerator protocol
- [ ] Day 6: TFT backtesting + baseline performance
  - Backtest TFT through walk-forward engine
  - Compare vs all 5 benchmarks
  - Regime-conditional performance analysis
  - Document when TFT works and when it fails
- [ ] Day 7: Week 1 review + refactor
  - Code quality review
  - End-to-end pipeline validation
  - Identify fragile points, refactor
  - Update TODO.md with actual progress
- [ ] Day 8: MAML Expert (fast adaptation for regime transitions)
  - Model-Agnostic Meta-Learning for time series
  - Different market regimes as different "tasks"
  - Few-shot adaptation with 5-10 days of data
  - Implement SignalGenerator protocol
- [ ] Day 9: N-BEATS/N-HiTS Expert (structural decomposition)
  - Interpretable basis expansion
  - Trend/seasonality/residual decomposition
  - Compare N-BEATS vs N-HiTS, pick better performer
  - Implement SignalGenerator protocol
- [ ] Day 10: (RESERVED) 4th expert ONLY if Day 6/8/9 backtests reveal a specific gap
  - If all 3 experts fail in the same regime, design an expert for that regime
  - If no clear gap, use this day for tuning existing experts
  - Decision based on DATA, not architecture ambition
- [ ] Day 11: Gating Network
  - Takes regime features (from Day 3 PCA/clustering) as input
  - Outputs expert weights (soft routing)
  - Trains on walk-forward basis (same anti-lookahead rules)
- [ ] Day 12: Full MoE integration + ensemble backtest
  - All 3 experts + gating network end-to-end
  - Compare MoE vs individual experts vs naive equal-weight ensemble
  - MoE should outperform any single expert (if not, diagnose why)
- [ ] Day 13-14: Ablation studies + Sprint 1 hardening
  - Which experts contribute most to alpha?
  - Does gating actually help vs equal weighting?
  - Feature importance analysis (which features drive the gating network?)
  - Fix any issues found
 
### Sprint 1 Week 3-4 (Statistical Validation + Data Hardening)
- [ ] Day 15: Multiple hypothesis testing framework (Bonferroni, FDR control)
- [ ] Day 16: Monte Carlo permutation tests for alpha significance
- [ ] Day 17: Survivorship bias analysis
- [ ] Day 18: Transaction cost sensitivity analysis
- [ ] Day 19: Regime-conditional performance deep dive
- [ ] Day 20-21: Fix everything testing revealed, expand data sources (macro, sector ETFs, VIX term structure)
- [ ] Comprehensive test suite target: >80% coverage on core modules
- [ ] Sprint 1 retrospective document
 
### Sprint 1 Exit Criteria
- [ ] MoE with 3 experts generating signals
- [ ] Backtested over 2015-2024 with walk-forward validation
- [ ] Out-of-sample Sharpe > 0.8 (2020-2024 period)
- [ ] Statistical significance tests passing (p < 0.05)
- [ ] Beats momentum benchmark after transaction costs
- [ ] All validators passing, clean codebase
- [ ] Architecture documented with decision rationale
 
---
 
## SPRINT 2: AI Agent Layer + Strategy Execution
## Timeline: Apr 15 - May 15, 2026
 
### Week 1: Agent Infrastructure
- [ ] Agent framework design with multi-LLM support from day 1
  - Abstract agent interface (LLM-agnostic)
  - Support Claude, GPT-4, DeepSeek
  - Agent config specifies which LLM to use
  - Standardized input/output schema for all agents
- [ ] Agent evaluation framework (how to measure if an agent is "good")
- [ ] Filing Analyst Agent (Claude)
  - SEC 10-K/10-Q retrieval via EDGAR API
  - Extracts: forward-looking statement changes, risk factor diffs, revenue segment narrative shifts
  - Uses value investing analytical framework (margin of safety, moat detection)
  - Output: structured numerical signals for MoE feature consumption
- [ ] Earnings Call Agent (GPT-4)
  - Transcript retrieval pipeline
  - Behavioral analysis framework: tone shifts, hedging language, Q&A dynamics, guidance parsing
  - Output: quantified sentiment scores as features
- [ ] Macro Narrative Agent (DeepSeek)
  - Fed minutes / FOMC statement analysis
  - Regime classification framework: Fed policy stance, credit conditions, growth/inflation
  - Output: macro regime signals feeding gating network

### Sprint 2 Pre-Work (do on Apr 14, day before Sprint 2 starts)
- [ ] Create AGENTS.md with full agent design spec
- [ ] Design agent input/output schemas based on MoE feature requirements
- [ ] Choose which LLM for which agent based on cost/capability analysis

### Week 2: Devil's Advocate + Signal Integration
- [ ] Devil's Advocate Agent
  - Runs on whichever LLM the primary analyst did NOT use
  - Generates counter-thesis + confidence score for each primary signal
  - Counter-thesis feeds into disagreement scoring, NOT a debate
  - One round only, parallel execution with primary agents
- [ ] Disagreement scoring system
  - All agents analyze independently in parallel
  - Compute pairwise disagreement between agents
  - High disagreement = lower position size (quantitative, automatic)
  - Log all agent outputs + disagreement scores for analysis
- [ ] Integrate agent signals into MoE as features
  - Convert agent outputs to numerical feature vectors
  - Add to feature store alongside existing features
  - A/B backtest: MoE with vs without agent features
  - Quantify alpha contribution from agents
- [ ] Agent cost analysis (API spend per signal, optimize prompts for efficiency)
- [ ] Agent reliability testing (hallucination rate, consistency across runs)
 
### Week 3: Strategy Execution Layer
- [ ] Pairs trading engine
  - Cointegration testing across universe
  - Dynamic pair selection (pairs cointegrated NOW, not just historically)
  - Z-score entry/exit signals
  - MoE forecasts inform which pairs to trade
- [ ] Strategy selection logic
  - Given MoE output: directional vs pairs vs no-trade
  - Position sizing based on signal conviction + disagreement score
  - Reasoning logged for every decision
- [ ] Execution optimization
  - Transaction cost minimization
  - Order splitting for larger positions
  - Timing optimization (avoid trading at open/close)
- [ ] Black-Litterman portfolio construction
  - Combine MoE forecasts with market equilibrium
  - Produce optimal portfolio weights
  - Replace any quantum-inspired approaches
 
### Week 4: Risk Management + Circuit Breakers
- [ ] VaR and CVaR computation (historical + parametric)
- [ ] Position concentration limits
- [ ] Sector exposure caps
- [ ] Correlation monitoring (alert on portfolio correlation spikes)
- [ ] PCA-based risk decomposition (what factors drive our risk?)
- [ ] Circuit breakers (HARDCODED, non-overridable):
  - Max daily drawdown -> auto-flatten all positions
  - Max single-position size (% of portfolio)
  - Max sector exposure (% of portfolio)
  - Model disagreement detector -> reduce sizing when experts diverge
  - Volatility scaling -> auto-deleverage when vol spikes
- [ ] Stress test through historical crises (COVID Mar 2020, 2022 rate hikes, Aug 2024 vol spike)
- [ ] Sprint 2 retrospective
 
### Sprint 2 Exit Criteria
- [ ] 3 LLM agents + 1 devil's advocate processing real data
- [ ] Quantified alpha contribution from agent signals
- [ ] Strategy layer selecting trade expression type
- [ ] Risk management constraining all positions
- [ ] Circuit breakers tested against historical crises
- [ ] Full system backtest: data -> agents -> MoE -> strategy -> risk -> trades
- [ ] Out-of-sample Sharpe > 1.0 with agent signals integrated
 
---
 
## SPRINT 3: Robustness + Quantitative Monitoring
## Timeline: May 15 - Jun 15, 2026
 
### Quantitative Monitoring (Layer 5 -- ALL MATH, no LLMs)
- [ ] Rolling Sharpe per expert (detect when an expert is degrading)
- [ ] Retraining trigger (when rolling Sharpe drops below threshold, auto-retrain)
- [ ] Gating audit (is gating network routing correctly? Compare gated vs equal-weight)
- [ ] Feature importance drift detection (which features matter is changing)
- [ ] Out-of-sample performance degradation monitoring
- [ ] Overfitting detector (compare recent IS vs OOS performance ratio)
- [ ] System health monitor (latency, API failures, data staleness)
 
### Adversarial Testing
- [ ] Synthetic stress scenarios (correlation spikes, liquidity crises, flash crashes)
- [ ] What if data source is wrong? (adversarial data injection)
- [ ] What if an LLM API is down? (graceful degradation testing)
- [ ] System should work with any single component offline (reduced capability, not crash)
 
### Correlation-Based Crowding Check (replaces cut Crowding Risk Agent)
- [ ] Monitor portfolio return correlation vs momentum factor ETF (MTUM)
- [ ] If correlation > threshold for N days, flag potential crowding
- [ ] 20-line function, not an agent
 
### Sprint 3 Exit Criteria
- [ ] Monitoring catches degradation in backtested scenarios
- [ ] System survives all historical stress scenarios
- [ ] Graceful degradation when components fail
- [ ] Full system re-backtest with all layers
- [ ] Out-of-sample Sharpe > 1.0, Max Drawdown < 12%
 
---
 
## SPRINT 4: Live Paper Trading
## Timeline: Jun 15 - Jul 15, 2026
 
### Infrastructure
- [ ] Live data feed integration (delayed is fine for paper trading)
- [ ] Real-time feature computation pipeline
- [ ] Signal generation running on daily schedule (automated)
- [ ] Paper trading ledger tracking every hypothetical trade with timestamps
- [ ] Broker API integration (Alpaca Paper Trading or IBKR Paper)
 
### Dashboard (3 essential views ONLY -- add more only when needed)
- [ ] View 1: Portfolio P&L + positions + benchmark comparison
- [ ] View 2: Risk metrics (VaR, exposure breakdown, circuit breaker status)
- [ ] View 3: Agent activity log (what each agent decided + disagreement scores)
- [ ] Alert system (Slack/Discord notifications for trades, circuit breaker events)
 
### Live Debugging
- [ ] Compare live signals vs backtest expectations
- [ ] Debug discrepancies (there WILL be discrepancies)
- [ ] Identify and fix: data latency, feature timing, execution issues
- [ ] Formal paper trading track record start date established
 
### Sprint 4 Exit Criteria
- [ ] Paper trading running daily without manual intervention
- [ ] Dashboard showing real-time portfolio status
- [ ] Live performance within 20% of backtest expectations
- [ ] System uptime > 95% for past 2 weeks
- [ ] Track record clock is ticking
 
---
 
## SPRINT 5: Track Record + Refinement
## Timeline: Jul 15 - Aug 15, 2026
 
- [ ] 30+ days of paper trading track record accumulating
- [ ] Expand data sources (alternative data exploration, higher-frequency data)
- [ ] Agent memory store (track what agents have seen, detect changes over time)
  - NOTE: only build this AFTER agents have been running and you can identify what they need to remember
- [ ] Performance analysis: which components generate alpha?
- [ ] Strategy refinement based on live results
- [ ] Compare against hedge fund benchmarks with real track record data
 
### Sprint 5 Exit Criteria
- [ ] 30+ days of paper trading track record
- [ ] Track record competitive with hedge fund benchmarks
- [ ] Clear attribution of where alpha comes from
- [ ] At least one expanded data source integrated
 
---
 
## SPRINT 6: YC Preparation
## Timeline: Aug 15 - Sep 15, 2026
 
### Technical Polish
- [ ] Code review entire codebase
- [ ] Open-source non-alpha components (data pipeline, feature store, agent framework)
- [ ] Architecture documentation (clean diagrams)
- [ ] Security audit (API keys, data access)
 
### YC Application Materials
- [ ] 1-minute video (problem, solution, traction, why you)
- [ ] Written application (thesis, differentiation, track record, market size, roadmap)
- [ ] 10-12 slide investor deck
- [ ] Mock interviews (5+ rounds)
- [ ] Prepare for tough questions:
  - "Why won't Renaissance just do this?"
  - "How do you know the backtest isn't overfit?"
  - "You're 17, why should we trust you with capital?"
  - "What's your data edge?"
 
### Sprint 6 Exit Criteria
- [ ] YC application submitted (target: October 2026)
- [ ] 60+ days of paper trading track record
- [ ] Investor deck polished
- [ ] Video recorded
- [ ] Mock interviews completed
- [ ] Benchmark comparison: Meridian vs virattt/ai-hedge-fund vs SPY
 
---
 
## ARCHITECTURE IMPROVEMENTS BACKLOG
## (Integrate when relevant, not as separate projects)
 
### High Priority
- [ ] Multi-LLM support in agent framework (Claude, GPT-4, DeepSeek) -- Sprint 2 Week 1
- [ ] Feature importance tracking over time (changes by regime) -- Sprint 1 Day 13
- [ ] Historical S&P 500 constituents for survivorship-bias-free backtesting -- Sprint 1 Day 17
 
### Medium Priority
- [ ] Docker multi-service setup (for paper trading deployment) -- Sprint 4
- [ ] GPU training support for larger models -- if needed
- [ ] A/B testing framework for strategy variants -- Sprint 3
- [ ] Automated hyperparameter tuning per expert -- Sprint 3
 
### Low Priority (Nice to Have)
- [ ] International market support
- [ ] Options strategy support
- [ ] Real-time news feed processing
- [ ] Satellite/alternative data integration
- [ ] Mobile dashboard
 
---
 
## COMPETITIVE INTELLIGENCE
 
### virattt/ai-hedge-fund (45.7k stars -- our baseline "100m² house")
**What they have:**
- 12+ LLM investor persona agents (Buffett, Munger, Cathie Wood, etc.)
- LangGraph orchestration
- Web UI (React frontend)
- Docker support
- Multi-LLM support (OpenAI, Anthropic, Groq, DeepSeek, Ollama)
- 799 commits, 32 contributors
 
**What they DON'T have (our advantages):**
- No ML models (pure LLM prompting) -- we have MoE with 3 trained experts
- No persistent data storage -- we have DuckDB with full pipeline
- No feature engineering -- we have 50+ features with anti-lookahead
- No regime detection -- we have rolling PCA + HMM + K-Means
- No statistical validation -- we have walk-forward + significance tests
- No anti-lookahead protection -- we have dedicated validator suite
- No execution optimization -- we have transaction cost models
- No circuit breakers -- we have hardcoded non-overridable limits
- No paper trading track record -- we will have 60+ days
- No real risk management -- we have VaR/CVaR + PCA decomposition
 
**When we surpass them by dimension:**
- Data infrastructure: Day 1 ✓
- Feature engineering: Day 2 ✓
- Regime detection: Day 3 ✓
- Backtesting rigor: Day 4 ✓
- ML models: Day 5-12
- Statistical validation: Day 15-17
- LLM agents (better architecture): Sprint 2
- Risk management: Sprint 2
- Web UI: Sprint 4
- Track record: Sprint 5
 
---
 
## WHAT WE CUT (iron audit, March 16 2026)
 
| Cut | Reason | Replaced With |
|-----|--------|---------------|
| Quantum portfolio optimization | Marginal benefit at our scale, resume decoration | Black-Litterman (what actual funds use) |
| Adversarial Transformer (expert #4) | Unproven in financial time series, research frontier | 3 experts; add 4th only if data shows a gap |
| Crowding Risk Agent (LLM) | No institutional flow data to detect crowding properly | Correlation check vs momentum ETF (20 lines) |
| Multi-round agent debate | Same model arguing with itself is theatre, costly | Parallel multi-LLM + disagreement scoring |
| Agent memory (pre-built) | Premature; agents don't exist yet | Build in Sprint 5 after agents are running |
| 10-view dashboard | Product, not tool; half-built 10 views < working 3 views | 3 essential views, add only when needed |
| Investor persona names | Makes us look like virattt clone; "Buffett agent" = toy optics | Domain-expert names with explicit analytical frameworks |
 
### What we KEPT (modified from original cut list)
| Kept | Modification | Why |
|------|-------------|-----|
| Multi-LLM disagreement | NOT debate; parallel analysis, disagreement quantified | Different LLMs have genuinely different biases |
| Devil's advocate agent | Runs on DIFFERENT LLM than primary; one round only | Different model = different blind spots = real challenge |
 
---
 
## DECISION LOG
 
| Date | Decision | Rationale |
|------|----------|-----------|
| Day 1 | DuckDB over PostgreSQL | Columnar, fast analytics, embedded, great pandas integration. See docs/decisions/001_database_choice.md |
| Day 3 | HMM + K-Means ensemble for regimes | HMM captures temporal dynamics, K-Means provides stability. Ensemble hedges model risk. |
| Day 3 | Rolling PCA with monthly refit | Balances adaptivity vs stability. Daily refit too noisy, yearly too slow. |
| Day 3 (audit) | 3 MoE experts, not 4 | Adversarial Transformer is unproven. Let backtesting data tell us if we need a 4th. |
| Day 3 (audit) | Black-Litterman, not quantum | Classical, proven at institutional scale. Quantum adds complexity without demonstrated benefit at our asset count. |
| Day 3 (audit) | Quantitative monitoring, not LLM monitoring | Math catches problems reliably. LLMs arguing with themselves is not risk management. |
| Day 3 (audit) | Multi-LLM parallel analysis | Claude + GPT-4 + DeepSeek have genuinely different training data and biases. Disagreement between them is a meaningful signal. |
| Day 3 (audit) | Domain-expert agent names | "Filing Analyst with value investing framework" > "Warren Buffett Agent". Sounds like a fund, not a toy. |
| Day 4 | Walk-forward over simple train/test split | Multiple OOS periods across different regimes prevents single-period overfitting. |
| Day 4 | Signal at t, execute at t+1 | Prevents same-day lookahead bias; matches real-world execution latency. |
| Day 4 | Almgren-Chriss sqrt impact as default | Industry-standard market impact model; conservative assumptions. |
| Day 4 | SignalGenerator as Protocol class | Any model with fit()/predict() works; decouples model development from backtesting. |
 
---
 
## BUGS / TECH DEBT
(Add items here as they come up during development)
 
---
 
## DAILY NOTES
(Quick notes from each day's work)
 
### Day 1
- All phases completed, tests passing
- Used combined prompt (v2), no issues with model
 
### Day 2
- Feature engineering pipeline complete
- 50+ features operational
- Anti-lookahead tests passing
 
### Day 3
- PCA + regime clustering operational
- 4 regimes detected, transitions align with COVID + 2022 rate hikes
- Docker build working
- Iron audit completed: cut quantum, adversarial expert, debate, crowding agent
- Kept multi-LLM disagreement + devil's advocate (modified)

### Day 4
- Backtesting framework complete: walk-forward engine, 5 benchmarks, validators
- 49 new tests, all passing; 91 non-DuckDB tests total passing
- SignalGenerator protocol ready for Day 5 TFT implementation
- Anti-lookahead enforced: signal at t, execute at t+1; strict train/test windowing
