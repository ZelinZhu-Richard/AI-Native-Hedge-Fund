# Project Meridian: 6-Month Roadmap
## Target: YC Winter 2027 Application (October 2026)
## Start Date: March 15, 2026

---

## Executive Timeline

```
MAR 15 ──── APR 15 ──── MAY 15 ──── JUN 15 ──── JUL 15 ──── AUG 15 ──── SEP 15 ── OCT
Sprint 1     Sprint 2     Sprint 3     Sprint 4     Sprint 5     Sprint 6     YC App
Foundation   MoE Engine   AI Agents    Live System  Track Record Polish+Apply
+ Data       + Backtest   + Strategy   + Paper Trade + Refinement
```

---

## SPRINT 1: Foundation + Core Infrastructure (Mar 15 - Apr 15)

### Theme: "Build the factory before the product"

**Week 1 (Mar 15-21): Project Scaffolding + Data Pipeline**
- Day 1: Monorepo structure, config, DuckDB schema (use Day 1 prompt)
- Day 2: Feature engineering pipeline (50+ technical/cross-sectional features)
- Day 3: PCA + regime clustering (HMM on principal component space)
- Day 4: Backtesting framework (walk-forward, no lookahead, transaction costs)
- Day 5: TFT Expert -- first MoE expert operational
- Day 6: TFT backtest + regime-conditional performance analysis
- Day 7: Week 1 review, refactor anything fragile

**Week 2 (Mar 22-28): Remaining MoE Experts**
- Day 8: MAML Expert (fast adaptation for regime transitions)
- Day 9: N-BEATS/N-HiTS Expert (structural decomposition)
- Day 10: Adversarial Transformer Expert (robustness under distribution shift)
- Day 11: Gating Network (regime-aware expert routing)
- Day 12: Full MoE integration + ensemble backtest
- Day 13-14: MoE tuning, ablation studies (which experts help, which hurt?)

**Week 3 (Mar 29 - Apr 4): Backtesting Rigor**
- Day 15: Multiple hypothesis testing framework (Bonferroni, FDR control)
- Day 16: Monte Carlo permutation tests for alpha significance
- Day 17: Survivorship bias analysis (how much does universe selection matter?)
- Day 18: Transaction cost sensitivity analysis
- Day 19: Regime-conditional performance deep dive
- Day 20-21: Fix everything the testing revealed

**Week 4 (Apr 5-15): Data Quality + Infrastructure Hardening**
- Expand data sources beyond yfinance (macro data, sector ETFs, VIX term structure)
- Data validation hardening (edge cases from real data)
- Feature store versioning (reproducible feature sets)
- Comprehensive test suite (>80% coverage on core modules)
- Sprint 1 retrospective document

### Sprint 1 Exit Criteria
- [ ] MoE with 4 experts generating signals
- [ ] Backtested over 2015-2024 with walk-forward validation
- [ ] Out-of-sample Sharpe > 0.8 (2020-2024 period)
- [ ] Statistical significance tests passing (p < 0.05)
- [ ] All tests passing, clean codebase
- [ ] Architecture documented with decision rationale

---

## SPRINT 2: AI Agent Layer + Strategy Execution (Apr 15 - May 15)

### Theme: "Where the AI-native edge actually lives"

**Week 1 (Apr 15-21): LLM Agent Infrastructure**
- Agent framework: standardized input/output schema for all agents
- Agent evaluation framework (how do you test if an agent is "good"?)
- Filing Analyst Agent (SEC 10-K/10-Q processing via EDGAR)
- Earnings Call Agent (transcript analysis, tone shift detection)
- Macro Narrative Agent (Fed minutes, macro regime classification)

**Week 2 (Apr 22-28): Agent Signal Integration**
- Convert agent outputs to numerical features for MoE
- Feature importance analysis: do LLM signals add alpha?
- A/B backtest: MoE with vs without LLM features
- Agent reliability testing (do they hallucinate? how often?)
- Agent cost analysis (API spend per signal, optimize prompt efficiency)

**Week 3 (Apr 29 - May 5): Strategy Execution Layer**
- Pairs trading engine (cointegration, dynamic pair selection)
- Strategy Selection Agent (directional vs pairs vs no-trade decisions)
- Execution optimization (transaction cost minimization)
- Position sizing framework (Kelly criterion variant, signal conviction scaling)

**Week 4 (May 6-15): Risk Management + Circuit Breakers**
- VaR/CVaR computation (historical + parametric)
- PCA risk decomposition
- Circuit breakers (hardcoded, non-overridable):
  - Max daily drawdown -> flatten
  - Position concentration limits
  - Sector exposure caps
  - Model disagreement -> reduce sizing
  - Volatility scaling -> auto-deleverage
- Stress test through historical crises (COVID, 2022 rates, Aug 2024)
- Sprint 2 retrospective

### Sprint 2 Exit Criteria
- [ ] LLM agents processing real filings and earnings calls
- [ ] Quantified alpha contribution from LLM signals
- [ ] Strategy layer selecting trade expression type
- [ ] Risk management constraining all positions
- [ ] Circuit breakers tested against historical crises
- [ ] Full system backtest: data -> agents -> MoE -> strategy -> risk -> trades
- [ ] Out-of-sample Sharpe > 1.0 with LLM signals integrated

---

## SPRINT 3: Red Team + Meta-Monitor + Robustness (May 15 - Jun 15)

### Theme: "Try to break it before the market does"

**Week 1 (May 15-21): Layer 0 -- Red Team**
- Devil's Advocate Agent (argues against every trade)
- Overfitting Detector (rolling out-of-sample degradation monitoring)
- Regime Uncertainty Quantifier (when gating network is confused, size down)
- Crowding Risk Agent (are we in the same trades as everyone else?)

**Week 2 (May 22-28): Layer 5 -- Meta-Monitor**
- Performance Attribution Agent (WHY did we make/lose money?)
- Model Decay Detector (rolling Sharpe per expert, retraining triggers)
- Gating Audit (is the gating network routing correctly?)
- System Health Monitor (latency, API failures, data staleness)

**Week 3 (May 29 - Jun 4): Adversarial Testing**
- Synthetic stress scenarios (correlation spikes, liquidity crises, flash crashes)
- Adversarial data injection (what if our data source is wrong?)
- Agent failure modes (what happens when Claude API is down?)
- Graceful degradation testing (system should work with any single component offline)

**Week 4 (Jun 5-15): Quantum-Inspired Portfolio Optimization**
- Research: tensor networks for portfolio optimization
- Implement quantum-inspired sampling for efficient frontier computation
- Compare vs classical mean-variance and Black-Litterman
- Integrate as portfolio construction step between strategy and risk layers
- Sprint 3 retrospective

### Sprint 3 Exit Criteria
- [ ] Red Team layer catching overfitting and bad trades
- [ ] Meta-Monitor tracking system health and model decay
- [ ] System survives all historical stress scenarios
- [ ] Graceful degradation when components fail
- [ ] Quantum-inspired portfolio optimization integrated
- [ ] Full system re-backtest with all layers
- [ ] Out-of-sample Sharpe > 1.0, Max Drawdown < 12%

---

## SPRINT 4: Live Paper Trading (Jun 15 - Jul 15)

### Theme: "Theory meets reality"

**Week 1 (Jun 15-21): Paper Trading Infrastructure**
- Live data feed integration (delayed is fine for paper trading)
- Real-time feature computation pipeline
- Signal generation running on daily schedule (automated)
- Paper trading ledger (track every hypothetical trade with timestamps)
- Broker API integration for paper trading (Alpaca Paper Trading or IBKR Paper)

**Week 2 (Jun 22-28): Dashboard + Monitoring**
- Real-time portfolio dashboard:
  - Current positions, P&L, exposure breakdown
  - Agent activity log (what each agent decided and why)
  - Risk metrics (live VaR, circuit breaker status)
  - Performance vs benchmarks (SPY, HFR Index)
- Alert system (Slack/Discord notifications for trades, circuit breaker events)

**Week 3 (Jun 29 - Jul 5): Live Debugging + Iteration**
- First 2 weeks of paper trading results analysis
- Compare live signals vs backtest expectations
- Debug discrepancies (there WILL be discrepancies)
- Identify and fix issues: data latency, feature computation timing, etc.
- This week is dedicated to making the live system MATCH the backtest

**Week 4 (Jul 6-15): Optimization + Stability**
- Reduce API costs (optimize agent prompts, cache where possible)
- Improve system reliability (retry logic, failover, monitoring)
- Performance optimization (can we generate signals faster?)
- Begin tracking formal paper trading track record (this date matters for YC)
- Sprint 4 retrospective

### Sprint 4 Exit Criteria
- [ ] Paper trading running daily without manual intervention
- [ ] Dashboard showing real-time portfolio status
- [ ] Alert system notifying on trades and circuit breaker events
- [ ] Live performance within 20% of backtest expectations
- [ ] System uptime > 95% for the past 2 weeks
- [ ] Formal track record start date established

---

## SPRINT 5: Track Record + Advanced Features (Jul 15 - Aug 15)

### Theme: "Let the results speak"

**Week 1-2 (Jul 15-28): Track Record Accumulation + Data Expansion**
- System running daily, accumulating paper trading track record
- Expand data sources:
  - Alternative data exploration (web traffic, satellite, social sentiment)
  - Higher-frequency data (intraday for execution optimization)
  - International markets (if thesis applies cross-market, that's a bigger TAM)
- Additional strategy types based on what's working

**Week 3 (Jul 29 - Aug 4): Advanced Agent Capabilities**
- Multi-agent debate system (agents argue before generating consensus signal)
- Agent memory (track what they've seen before, detect changes over time)
- Cross-asset reasoning (agent notices "oil earnings bad" -> implications for airlines)
- Agent self-evaluation (confidence calibration -- are they overconfident?)

**Week 4 (Aug 5-15): Performance Analysis + Refinement**
- 30+ days of paper trading track record analysis
- Detailed performance attribution (which components generate alpha?)
- Strategy refinement based on live results
- Compare against hedge fund benchmarks with real track record data
- Sprint 5 retrospective

### Sprint 5 Exit Criteria
- [ ] 30+ days of paper trading track record
- [ ] Track record competitive with hedge fund benchmarks
- [ ] At least one expanded data source integrated
- [ ] Multi-agent capabilities operational
- [ ] Clear attribution of where alpha comes from
- [ ] Identified top 3 improvements for Sprint 6

---

## SPRINT 6: YC Preparation + Polish (Aug 15 - Sep 15)

### Theme: "Package the rocket ship"

**Week 1 (Aug 15-21): Technical Polish**
- Code review entire codebase (refactor, document, clean up)
- Open-source preparation (if strategy details can be separated from alpha-generating code)
- Architecture documentation (clean diagrams, system design doc)
- Security audit (API keys, data access, no secrets in code)

**Week 2 (Aug 22-28): YC Application Materials**
- One-minute video (required by YC):
  - Problem: quant funds can't adapt to regime changes
  - Solution: AI-native MoE architecture with LLM agents
  - Traction: X days of paper trading, Sharpe of Y, alpha of Z
  - Why you: built this solo as a high schooler, previously built Apex Analysis
- Written application:
  - Clear thesis statement
  - Technical differentiation
  - Track record data
  - Market size (global hedge fund AUM: $4.5T+)
  - Roadmap: paper trading -> seed raise -> live trading

**Week 3 (Aug 29 - Sep 4): Investor Deck**
- 10-12 slide deck:
  1. Problem (regime changes break quant funds)
  2. Solution (AI-native MoE + LLM agents)
  3. How It Works (architecture diagram, one layer per slide)
  4. Results (backtest + live paper trading track record)
  5. Why Now (LLM capabilities just reached the threshold)
  6. Market Size ($4.5T+ hedge fund AUM)
  7. Competition (why existing funds can't replicate this easily)
  8. Business Model (2/20 fee structure, target AUM path)
  9. Team (you + advisors + plans for co-founder)
  10. Roadmap (seed raise -> live trading -> scale)
  11. Ask (YC investment + network for institutional introductions)

**Week 4 (Sep 5-15): Practice + Refine**
- Practice YC interview format (2 minutes to explain, then rapid-fire Q&A)
- Prepare for tough questions:
  - "Why won't Renaissance just do this?"
  - "How do you know the backtest isn't overfit?"
  - "You're 17, why should we trust you with capital?"
  - "What's your data edge?"
- Final system demo preparation (live paper trading walkthrough)
- Sprint 6 retrospective

### Sprint 6 Exit Criteria
- [ ] YC application submitted
- [ ] 60+ days of paper trading track record
- [ ] Investor deck polished
- [ ] 1-minute video recorded
- [ ] Mock interview practice completed (minimum 5 rounds)
- [ ] Codebase is open-source ready (or strategic portions are)

---

## POST-APPLICATION: Sep 15 - Dec 2026

While waiting for YC decision:
- Continue accumulating track record (every additional day strengthens your case)
- Explore seed funding from angel investors in quant finance
- Co-founder search (attend quant finance meetups, reach out to grad students)
- Legal: research fund formation, compliance requirements (SEC, NFA)
- Begin conversations with potential institutional data providers
- Consider: apply to other accelerators as backup (Entrepreneur First, Techstars)

---

## KEY MILESTONES SUMMARY

| Date | Milestone | Why It Matters |
|---|---|---|
| Apr 15 | MoE backtested, Sharpe > 0.8 | Core engine validated |
| May 15 | LLM agents adding alpha | AI-native edge proven |
| Jun 15 | Red team + stress tested | Institutional-grade robustness |
| Jul 15 | Paper trading live | Theory -> reality transition |
| Aug 15 | 30-day track record | Real performance data |
| Sep 15 | YC app submitted, 60-day track record | The pitch is backed by evidence |
| Oct 2026 | YC application deadline | Ship date |

---

## RISK REGISTER

| Risk | Impact | Mitigation |
|---|---|---|
| MoE doesn't outperform single models | High | Ablation studies, expert redesign |
| LLM signals don't add alpha | Medium | Fall back to pure quant (still viable) |
| Live performance << backtest | High | Sprint 4 dedicated to gap analysis |
| Claude API costs too high | Medium | Prompt optimization, caching, batching |
| Solo builder burnout | High | Sustainable pace, 30-day sprints with breaks |
| No co-founder for YC | Medium | Strong advisor network compensates |
| Overfitting all strategies | Critical | Red team layer, multiple significance tests |
| YC rejects application | Medium | Track record still exists, apply elsewhere |

---

## BUDGET ESTIMATE (6 months)

| Item | Monthly | 6-Month Total |
|---|---|---|
| Claude API (agent processing) | $50-150 | $300-900 |
| Market data (yfinance free, Alpha Vantage) | $0-50 | $0-300 |
| Cloud compute (training, if needed) | $0-100 | $0-600 |
| Alpaca Paper Trading API | $0 | $0 |
| Domain + hosting (dashboard) | $10-20 | $60-120 |
| **Total** | **$60-320** | **$360-1,920** |

This is buildable on a high school student's budget. That's part of the story.

---

## WHAT MAKES THIS YC-READY

1. **Technical depth**: MoE + LLM agents is genuinely novel, not just "GPT wrapper"
2. **Real results**: 60+ days of paper trading, not just backtests
3. **Solo execution**: A high schooler built this alone -- that's the "unreasonable founder" signal YC loves
4. **Clear thesis**: Not "AI for finance" but "regime transitions are mispriced and here's exactly how we exploit it"
5. **Big market**: Hedge fund industry is $4.5T+ AUM, even a tiny slice is massive
6. **Timing**: LLM capabilities just crossed the threshold for this to be possible
7. **Defensible moat**: Architecture + data pipeline + track record compound over time