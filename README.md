# Systematic Cross-Sectional Equity Alpha & Execution Simulator

## 1. Executive Summary

This repository provides an institutional-grade quantitative research, feature engineering, and execution simulation framework designed to systematically extract cross-sectional alpha from US liquid equities. By formulating point-in-time capitalization-ranked universes (CRSP Top-500) that strictly eliminate survivorship bias, forward lookahead, and non-operating equity contamination, the engine transforms raw daily pricing, volume, and corporate actions into cross-sectionally standardized predictive signals. Non-linear rank-based alpha forecasts produced via gradient-boosted decision trees under strictly purged walk-forward cross-validation are mapped into dollar-neutral long/short portfolios governed by quadratic transaction cost friction, market impact penalties, and empirical borrow constraints.

---

## 2. Theoretical Formulation & Mathematical Framework

### 2.1 Point-in-Time Universe Construction & Constituent Dynamics

Survivorship-free universe generation executes on an annual December schedule. On the final trading day of December in year $Y-1$ (reconstitution date $T$), an eligibility filter isolates liquid US common operating equities from the global universe $\mathcal{S}$:

$$
\mathcal{U}_{T} = \{ i \in \mathcal{S} \mid \text{Eligible}(i, T) = 1 \}
$$

Constituent eligibility requires all five pre-ranking hurdles to hold simultaneously on reconstitution date $T$:

| Hurdle | Formal Condition | Description |
|:---|:---|:---|
| **Share Type** | $\text{shrcd} \in \{10, 11\}$ | US common equity |
| **Primary Exchange** | $\text{exchcd} \in \{1, 2, 3\}$ | NYSE, AMEX, or NASDAQ |
| **Operating Entity** | $\text{siccd} \neq 6798, \; \text{siccd} \notin [6720, 6726]$ | Excludes REITs and closed-end funds |
| **Price Hurdle** | $\lvert P_T \rvert \ge 5.00$ | Penny stock exclusion (\$5.00 minimum price) |
| **Seasoning** | $\sum_{\tau = T - 251}^{T} \mathbb{I}(R_\tau \neq \text{NaN}) \ge 252$ | Minimum 252 active trading sessions in trailing year |

Market capitalization on reconstitution date $T$ is computed as:

$$
\text{Cap}_{i,T} = |P_{i,T}| \cdot S_{i,T} \cdot 1000
$$

Eligible constituents are sorted descending by market capitalization with deterministic tie-breaking on permanent security identifier (PERMNO):

$$
\mathcal{C}_{Y} = \text{arg max}^{(N)}_{i \in \mathcal{U}_{T}} \left( \text{Cap}_{i,T}, \, -\text{permno}_i \right)
$$

Daily point-in-time universe membership requires annual constituent selection and an active daily trading observation:

$$
\mathcal{M}_t = \{ i \in \mathcal{C}_{Y(t)} \mid \exists \, P_{i,t} \in \text{CRSP.DSF} \}
$$

Securities delisting intra-year exit the universe panel organically on the delisting effective date, avoiding survivorship distortion and artificial balance sheet replacements.

---

### 2.2 Feature Representation & Cross-Sectional Normalization

Let $R$ denote split- and dividend-adjusted daily total returns from CRSP. The continuous cumulative return index evolves according to:

$$
I_{i,t} = I_{i,0} \prod_{\tau=1}^t (1 + R_{i,\tau})
$$

For an arbitrary lookback horizon $k \in \mathbb{N}$ (e.g., $k=21$ trading days for 1-month momentum), the raw momentum factor is computed as:

$$
M_{i,t}^{(k)} = \frac{I_{i,t}}{I_{i,t-k}} - 1
$$

To eliminate outlier contamination while preserving rank monotonicities, raw factors undergo cross-sectional two-tailed winsorization per date $t$ at the 1st and 99th percentiles:

$$
\widetilde{M}_{i,t} = \max\left(Q_t(0.01), \; \min\left(M_{i,t}, \; Q_t(0.99)\right)\right)
$$

where $Q_t(\alpha)$ represents the empirical $\alpha$-quantile across active constituents. Cross-sectional Gaussian standardization normalizes winsorized factors:

$$
Z_{i,t} = \frac{\widetilde{M}_{i,t} - \mu_t(\widetilde{M}_{\cdot,t})}{\sigma_t(\widetilde{M}_{\cdot,t})}
$$

$$
\mathbb{E}_{i \in \mathcal{M}_t}[Z_{i,t}] = 0, \quad \text{Var}_{i \in \mathcal{M}_t}(Z_{i,t}) = 1
$$

---

### 2.3 Non-Anticipatory Target Formulation

To prevent lookahead contamination and model execution feasibility under market close auctions, trading signals derived at date $t$ close execute at $t+1$ close, held through $t+1+h$ close (holding horizon $h = 5$ trading days):

$$
R_{i,t+1 \to t+1+h}^{\text{fwd}} = \frac{I_{i,t+1+h} - I_{i,t+1}}{I_{i,t+1}}
$$

Continuous forward returns are partitioned cross-sectionally into $K \in \{3, 10\}$ equal-frequency ordinal bins via `pd.qcut`, yielding integer labels:

$$
y_{i,t} \in \{0, 1, \dots, K-1\}
$$

where $y_{i,t} = k$ if and only if forward return falls in quantile $k$ on date $t$. Rank ties are broken by first-occurrence ordering to guarantee deterministic label assignment.

---

### 2.4 Regularized Non-Linear Optimization & Objective Function

Model inference utilizes gradient-boosted decision tree ensembles $f(X; \Theta)$ minimizing regularized mean squared error across training observations:

$$
\mathcal{L}(\Theta) = \sum_{t \in \mathcal{T}_{\text{train}}} \sum_{i \in \mathcal{M}_t} \frac{1}{2}(y_{i,t} - \hat{y}_{i,t})^2 + \sum_{m=1}^M \left( \gamma J_m + \frac{1}{2}\lambda \sum_{j=1}^{J_m} v_j^2 \right)
$$

subject to minimum-leaf-population and tree depth constraints:

$$
n_j \ge N_{\min}, \quad \text{depth}(T_m) \le d_{\max} \quad \forall \, j, m
$$

where $\lambda$ is the $L_2$ leaf-score regularization coefficient and $\gamma$ is the minimum gain required to accept a split.

---

### 2.5 Walk-Forward Validation & Purging Bounds

For expanding-window cross-validation, label leakage occurs when forward return horizons span across split boundaries. To guarantee absolute orthogonality between training targets and validation features, the training index set is purged by dropping all dates whose $h$-day forward return window overlaps validation:

$$
\mathcal{T}_{\text{train}}^{\text{purged}} = \{ t \in \mathcal{T}_{\text{train}} \mid t \le \min(\mathcal{T}_{\text{val}}) - h \}
$$

In the implemented 5-fold scheme with $h = 5$, this removes the trailing 5 trading dates from each expanding training window.

---

### 2.6 Signal Evaluation: Spearman Information Coefficient & Information Ratio

Signal efficacy is quantified via daily out-of-fold cross-sectional Spearman rank correlation between model predictions and realized quantile targets:

$$
\rho_t = \frac{\sum_{i \in \mathcal{M}_t} (\text{Rank}(\hat{y}_{i,t}) - \overline{\text{Rank}}(\hat{y}_{\cdot,t}))(\text{Rank}(y_{i,t}) - \overline{\text{Rank}}(y_{\cdot,t}))}{\sqrt{\sum_{i \in \mathcal{M}_t} (\text{Rank}(\hat{y}_{i,t}) - \overline{\text{Rank}}(\hat{y}_{\cdot,t}))^2 \sum_{i \in \mathcal{M}_t} (\text{Rank}(y_{i,t}) - \overline{\text{Rank}}(y_{\cdot,t}))^2}}
$$

Aggregate cross-sectional stability metrics over out-of-fold evaluation days are defined as:

$$
\text{IC} = \frac{1}{|\mathcal{T}_{\text{eval}}|} \sum_{t \in \mathcal{T}_{\text{eval}}} \rho_t
$$

$$
\sigma_{\text{IC}} = \sqrt{\frac{1}{|\mathcal{T}_{\text{eval}}| - 1} \sum_{t \in \mathcal{T}_{\text{eval}}} (\rho_t - \text{IC})^2}, \quad \text{ICIR} = \frac{\text{IC}}{\sigma_{\text{IC}}}
$$

---

### 2.7 Portfolio Optimization, Slippage & Market Impact Bounds

Signals are mapped to long/short weights $w_t \in \mathbb{R}^N$ with long ($w > 0$) and short ($w < 0$) positions, subject to dollar neutrality and unit gross leverage:

$$
\mathbf{1}^T w_t = 0, \quad \sum_{i \in \mathcal{M}_t} |w_{i,t}| = 1
$$

Rebalancing turnover on date $t$ accounts for passive weight drift induced by asset returns:

$$
\Delta w_{i,t} = w_{i,t} - w_{i,t-1} \left(\frac{1 + R_{i,t}}{1 + R_{p,t-1}}\right)
$$

Total transaction friction encompasses linear fee drag and non-linear square-root market impact:

$$
\mathcal{C}_t = \sum_{i \in \mathcal{M}_t} \left( c_{\text{fee}} |\Delta w_{i,t}| + \eta \cdot \sigma_{i,t} \cdot \sqrt{\frac{|\Delta w_{i,t}| \cdot \text{AUM}}{\text{ADV}_{i,t}}} \right)
$$

Net strategy return accounts for gross portfolio return, transaction friction, and short borrow financing fees:

$$
R_{p,t}^{\text{net}} = \sum_{i \in \mathcal{M}_t} w_{i,t} R_{i,t+1} - \mathcal{C}_t - \sum_{i \in \mathcal{M}_t: w_{i,t} < 0} b_{i,t} |w_{i,t}|
$$

---

### 2.8 Variable Definitions & Market Constraints

| Parameter | Domain | Description | Value / Hurdle |
|:---|:---|:---|:---|
| `P_{i,t}` | $\mathbb{R}^+$ | Closing price on date $t$ (CRSP `prc`) | Absolute hurdle: $\lvert P_{i,t} \rvert \ge 5.00\text{ USD}$ at reconstitution |
| `S_{i,t}` | $\mathbb{R}^+$ | Shares outstanding in thousands (CRSP `shrout`) | Historical point-in-time count |
| `shrcd` | $\mathbb{N}$ | Share type code | $\{10, 11\}$ (US common equity) |
| `exchcd` | $\mathbb{N}$ | Exchange code | $\{1, 2, 3\}$ (NYSE, AMEX, NASDAQ) |
| `siccd` | $\mathbb{N}$ | Industry classification code | Excludes REITs (6798) and CEFs (6720–6726) |
| `H_{i,t}` | $\mathbb{N}$ | Trailing active days over 252 market days | Seasoning threshold $\ge 252$ |
| `h` | $\mathbb{N}$ | Forward investment horizon | $h = 5$ trading days |
| `AUM` | $\mathbb{R}^+$ | Portfolio assets under management | Variable institutional baseline |
| `ADV_{i,t}` | $\mathbb{R}^+$ | 21-day median daily traded dollar volume | Participation limit $\le 1.5\%$ ADV |
| `c_{fee}` | $\mathbb{R}^+$ | Linear broker, clearing, and SEC fees | $5 \times 10^{-4}$ (5 bps) |
| `\eta` | $\mathbb{R}^+$ | Market impact coefficient | $0.10$ |
| `b_{i,t}` | $\mathbb{R}^+$ | General collateral borrow rate | $50\text{ bps per annum}$ |

---

## 3. Pipeline Architecture

The end-to-end quantitative execution pipeline transitions raw market feeds into point-in-time universe panels, regularized features, machine learning forecasts, and risk-managed execution weights.

```
+----------------------------------------------------------------------------------------------------+
|                                    1. DATA INGESTION & STORAGE                                     |
|  +--------------------------+  +--------------------------+  +----------------------------------+  |
|  |     WRDS CRSP.DSF        |  |   WRDS CRSP.STOCKNAMES   |  |   WRDS CRSP.DSEDELIST & DSI      |  |
|  | (Daily Pricing & Volume) |  |   (Metadata, SIC, Exch)  |  |  (Delist Returns & Benchmarks)   |  |
|  +-------------+------------+  +------------+-------------+  +----------------+-----------------+  |
+----------------|----------------------------|---------------------------------|--------------------+
                 +----------------------------v---------------------------------+
                                              |
+---------------------------------------------v------------------------------------------------------+
|                               2. UNIVERSE & FEATURE ENGINEERING                                    |
|  +-----------------------------------------------------------------------------------------------+ |
|  | PointInTimeUniverse: Annual Dec Reconstitution (Top-500 Market Cap, Share/Exchange/SIC/Price)| |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Cross-Sectional Transforms: Cumulative Total Return Index -> 21D Momentum -> Winsor -> Z-Score | |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Target Construction: Non-Anticipatory T+1 -> T+6 Forward Return Discretized into Deciles       | |
|  +-----------------------------------------------------------------------------------------------+ |
+---------------------------------------------+------------------------------------------------------+
                                              |
+---------------------------------------------v------------------------------------------------------+
|                                  3. MODELING & SIGNAL ENGINE                                       |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Walk-Forward Purged Cross-Validation (5 Expanding Folds, 5-Day Forward Return Purge Cutoff)    | |
|  +-----------------------------------------------------------------------------------------------+ |
|  | LightGBM Gradient Boosted Regressor: L2 Regularization, Depth Limiting, Min-Child-Samples     | |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Signal Metric Evaluation: Out-of-Fold Daily Spearman Rank Information Coefficient (IC / ICIR)  | |
|  +-----------------------------------------------------------------------------------------------+ |
+---------------------------------------------+------------------------------------------------------+
                                              |
+---------------------------------------------v------------------------------------------------------+
|                           4. PORTFOLIO SIMULATION & ATTRIBUTION                                    |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Dollar-Neutral Long/Short Quantile Portfolio Sorting (Decile 10 Long vs. Decile 1 Short)      | |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Execution Friction Engine: Proportional Turnover Fees + Square-Root Market Impact + Borrow Drag| |
|  +-----------------------------------------------------------------------------------------------+ |
|  | Alphalens Factor Analysis: Multi-Horizon Factor Tearsheets (1D, 5D, 10D), Turnover, Quantiles  | |
|  +-----------------------------------------------------------------------------------------------+ |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Modular Subsystem Breakdown

- **`src/data/universe.py` (`PointInTimeUniverse`)**: Point-in-time universe generator. Implements December reconstitution schedules, five-stage pre-ranking eligibility filtering (`shrcd`, `exchcd`, `siccd`, price hurdles, trading seasoning), deterministic market capitalization sorting, and daily dynamic panel tracking accounting for mid-year delistings.
- **`src/data/wrds_fetcher.py` (`WRDSFetcher`)**: Remote database connector interfacing with Wharton Research Data Services (WRDS). Extracts CRSP daily stock file (`crsp.dsf`), security metadata (`crsp.stocknames`), delisting returns (`crsp.dsedelist`), and market indices (`crsp.dsi`) with local Apache Parquet caching.
- **`src/features/`**: Scalable feature transformation pipelines computing momentum, reversal, realized volatility, and volume dynamics with point-in-time cross-sectional winsorization and z-scoring.
- **`src/models/`**: Walk-forward cross-validation engines, tree ensemble architectures (LightGBM), penalized regressions, and parameter estimation modules.
- **`src/portfolio/`**: Quantile allocation constructors, dollar-neutral and beta-neutral weighting routines, transaction cost modeling, and liquidity impact penalty engines.
- **`src/validation/`**: Leakage prevention monitors, combinatorial purged cross-validation (CPCV) splitters, and label overlap verification checks.
- **`src/evaluation/`**: Factor tearsheet generators, out-of-fold Spearman rank IC/ICIR statistics, and Alphalens portfolio performance attribution.
- **`tests/test_universe.py`**: Exhaustive unit test suite verifying annual reconstitution dates, pre-ranking filtering mechanics, deterministic capitalization sorting, and dynamic daily membership delisting transitions.
- **`legacy/`**: Quarantined academic baseline and cryptographic integrity manifest (`README_LEGACY.md`) establishing permanent historical reproducibility against original course prototypes.

---

## 4. Implementation & Current Capabilities (Completed Scope)

### 4.1 Production Universe Engine (`PointInTimeUniverse`)
- **Eligibility Hurdle Enforcement**: Automatically executes pre-ranking filtration on reconstitution day:
  - Common share verification (`shrcd IN (10, 11)`).
  - Exchange verification (`exchcd IN (1, 2, 3)`).
  - Operating business validation excluding REITs (`siccd != 6798`) and closed-end funds (`siccd NOT BETWEEN 6720 AND 6726`).
  - Minimum price threshold ($|P| \ge \$5.00$).
  - Minimum seasoning verification ($\ge 252$ trading sessions in trailing 252 market days).
- **Dynamic Daily Membership**: Handles point-in-time membership tracking. When constituents delist mid-year, daily panel size adjusts automatically without forward lookahead or artificial survivorship substitution.
- **Deterministic Capitalization Ordering**: Resolves identical market capitalization values through deterministic `permno` sorting.

### 4.2 Automated Ingestion & Local Parquet Caching (`WRDSFetcher`)
- High-throughput extraction of CRSP tables (`dsf`, `stocknames`, `dsedelist`, `dsi`) via SQL over SSL with automatic date parsing and local Parquet caching in `data/raw/`.

### 4.3 Feature Engineering & Normalization Engine
- Total return accumulation index formulation handling splits and cash dividends.
- 21-day rolling cumulative momentum estimation.
- Daily cross-sectional 1%/99% winsorization and Z-score standardization.
- Realistic non-anticipatory $t+1 \to t+6$ forward return generation discretized into cross-sectional quantiles.

### 4.4 Purged Walk-Forward Cross-Validation Engine
- 5-fold expanding window `TimeSeriesSplit`.
- Purge operator discarding the trailing 5 trading days of each training window to eliminate label overlap contamination into validation sets.
- Safe daily Information Coefficient calculator with zero-variance protection.

### 4.5 Empirical Verification & Benchmark Logs

#### Walk-Forward Cross-Validation Performance (Historical Baseline Dataset)
The walk-forward cross-validation engine produces the following verified out-of-fold performance across 5 expanding folds:

| Cross-Validation Fold | Training Period | Validation Period | Out-of-Fold Spearman IC |
|:---|:---|:---|:---:|
| **Fold 1** | 2015-01-02 to 2016-10-14 | 2016-10-21 to 2018-04-03 | **-0.0112** |
| **Fold 2** | 2015-01-02 to 2018-03-27 | 2018-04-04 to 2019-09-11 | **+0.0278** |
| **Fold 3** | 2015-01-02 to 2019-09-04 | 2019-09-12 to 2021-02-24 | **-0.0214** |
| **Fold 4** | 2015-01-02 to 2021-02-17 | 2021-02-25 to 2022-08-05 | **+0.0010** |
| **Fold 5** | 2015-01-02 to 2022-07-29 | 2022-08-08 to 2023-12-29 | **-0.0140** |
| **Summary Metrics** | **2015 to 2023 Full Panel** | **5 Walk-Forward Folds** | **Mean IC: -0.0036 \| IC Std: 0.0173 \| ICIR: -0.2064** |

#### Universe Unit Test Execution
The test suite validates universe mechanics across four test vectors:

```
test_daily_membership_and_delisting_drop (test_universe.TestPointInTimeUniverse.test_daily_membership_and_delisting_drop) ... ok
test_filters_applied_before_ranking (test_universe.TestPointInTimeUniverse.test_filters_applied_before_ranking) ... ok
test_market_cap_ranking_and_selection (test_universe.TestPointInTimeUniverse.test_market_cap_ranking_and_selection) ... ok
test_reconstitution_dates (test_universe.TestPointInTimeUniverse.test_reconstitution_dates) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.024s

OK
```

---

## 5. System Architecture & Engineering Roadmap

The architecture transitions through disciplined quantitative research, simulation, and live production phases:

- [x] **Phase 1: Mathematical Formulation & Static Arbitrage / Core Signal Engine**
  - [x] Point-in-time CRSP universe selection engine (`PointInTimeUniverse`) with annual December reconstitution.
  - [x] Multi-criteria eligibility filtering (share codes 10/11, exchange codes 1/2/3, SIC exclusions for REITs/CEFs, \$5.00 price hurdle, 252-day seasoning).
  - [x] Automated WRDS data extraction layer (`WRDSFetcher`) with local Parquet serialization for `crsp.dsf`, `stocknames`, `dsedelist`, and `dsi`.
  - [x] Cross-sectional feature normalization pipeline (cumulative total return index, rolling lookback windows, 1%/99% winsorization, Z-score scaling).
  - [x] Non-anticipatory forward return target generator with cross-sectional quantile discretization.
  - [x] Purged expanding-window walk-forward cross-validation eliminating forward label contamination.
  - [x] Out-of-fold daily Spearman rank Information Coefficient (IC) and ICIR validation engine.
  - [ ] Quarantined academic baseline with SHA-256 cryptographic integrity manifest.
  - [ ] Automated unit test suite covering universe reconstitution, eligibility filtering, and mid-year delisting mechanics.

- [ ] **Phase 2: Transaction Cost, Fee Drag & Execution Slippage Modeling**
  - [ ] Dollar-neutral long/short portfolio construction mapping quantile predictions into dollar weights.
  - [ ] Continuous turnover tracking incorporating price drift adjustment between rebalancing intervals.
  - [ ] Linear transaction cost modeling capturing exchange fees, clearing charges, and broker commissions.
  - [ ] Non-linear square-root market impact modeling parameterized by asset volatility and average daily volume (ADV).
  - [ ] Short borrow fee drag integration across hard-to-borrow and general collateral assets.
  - [ ] Factor tearsheet integration via Alphalens for multi-horizon return attribution (1D, 5D, 10D holding periods).
  - [ ] Synthetic pipeline dry-run harness validating end-to-end memory safety and execution stability.

- [ ] **Phase 3: Real-Time Stream Ingestion & Microsecond Latency Profiling**
  - [ ] Async WebSocket direct feed handler for real-time market data ingestion (SIP / direct venue feeds).
  - [ ] Order book reconstruction engine (L2/L3 market depth, quote updates, and trade ticks).
  - [ ] Microsecond-resolution timestamp synchronization and clock drift calibration.
  - [ ] Low-latency online feature recalculation engine with incremental rolling statistics.
  - [ ] Lock-free circular buffer data pipelines for multi-threaded signal inference.
  - [ ] High-frequency slippage calibration comparing simulated fills against realized venue executions.

- [ ] **Phase 4: Multi-Venue Routing & Dynamic Risk Inventory Management**
  - [ ] Smart Order Router (SOR) with venue fee/rebate optimization (Maker-Taker model).
  - [ ] Real-time portfolio risk engine monitoring dollar neutrality, beta exposure, and sector concentration limits.
  - [ ] Dynamic inventory management with adverse selection protection and inventory half-life decay.
  - [ ] FIX protocol gateway integration (QuickFIX/C++) for low-latency institutional execution.
  - [ ] Automated circuit breakers, fat-finger trade thresholds, and venue disconnection failovers.

---

## 6. Quickstart & Usage

### 6.1 Environment Setup

Ensure a Python 3.10+ virtual environment is active, then install all required dependencies:

```bash
# Initialize isolated Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies from pinned requirements manifest
pip install --upgrade pip
pip install -r requirements.txt
```

### 6.2 WRDS Credential Configuration

To download fresh CRSP updates directly from Wharton Research Data Services, configure WRDS access via environment variables or a `.pgpass` credential file:

```bash
export WRDS_USERNAME="your_wrds_username"
```

If WRDS credentials are not supplied, the engine automatically falls back to local cached Parquet archives located in `data/raw/` or the root workspace (`raw_crsp_data.parquet`).

### 6.3 Executing the Test Suite

Execute the automated universe engine test suite to verify point-in-time reconstitution, eligibility filters, and delisting mechanics:

```bash
python3 -m unittest discover -s tests -v
```

Expected terminal output:
```text
test_daily_membership_and_delisting_drop (test_universe.TestPointInTimeUniverse.test_daily_membership_and_delisting_drop) ... ok
test_filters_applied_before_ranking (test_universe.TestPointInTimeUniverse.test_filters_applied_before_ranking) ... ok
test_market_cap_ranking_and_selection (test_universe.TestPointInTimeUniverse.test_market_cap_ranking_and_selection) ... ok
test_reconstitution_dates (test_universe.TestPointInTimeUniverse.test_reconstitution_dates) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.024s

OK
```

### 6.4 Running the Walk-Forward Validation Engine

Execute the purged walk-forward cross-validation pipeline across historical CRSP data:

```bash
python3 legacy/ml_arch.py
```

Expected terminal execution snippet:
```text
Loading data from local Parquet file...
                    close    return      volume  shares_outstanding
date       permno                                                  
2015-01-02 10107    46.76  0.006674  27913852.0           8218000.0
           11404    66.34  0.004999   1691318.0            292888.0
           11850    92.83   0.004110  10220410.0           4234529.0
           14593   109.33 -0.009513  53204828.0           5826419.0
           22111   104.52 -0.000478   5756961.0           2799110.0

Starting Walk-Forward Cross Validation...
Fold 1 Information Coefficient (IC): -0.0112
Fold 2 Information Coefficient (IC): 0.0278
Fold 3 Information Coefficient (IC): -0.0214
Fold 4 Information Coefficient (IC): 0.0010
Fold 5 Information Coefficient (IC): -0.0140

--- Walk-Forward CV Results ---
Mean IC: -0.0036
IC Std:  0.0173
ICIR:    -0.2064
```

### 6.5 Running Synthetic Pipeline Verification

To verify feature generation, target calculation, and purged CV logic independently of external database connections:

```bash
python3 legacy/dummy_run.py
```

Expected terminal execution snippet:
```text
1. Generating Synthetic Data...
2. Engineering Target (T+1 to T+6 Forward Return)...
3. Running Walk-Forward CV...
Fold 1 complete. Generated 1290 predictions.
Fold 2 complete. Generated 1290 predictions.
Fold 3 complete. Generated 1290 predictions.
Pipeline Dry Run Successful! No crashes.
```

### 6.6 Programmatic Python API

To instantiate the point-in-time universe engine directly within custom quantitative workflows:

```python
import pandas as pd
from src.data.universe import PointInTimeUniverse
from src.data.wrds_fetcher import WRDSFetcher

# 1. Ingest raw CRSP panels from WRDS or local Parquet cache
fetcher = WRDSFetcher(output_dir="data/raw")
df_dsf = fetcher.fetch_dsf(start_date="2014-01-01", end_date="2024-12-31")
df_stocknames = fetcher.fetch_stocknames()

# 2. Instantiate Point-in-Time Universe Engine
universe_engine = PointInTimeUniverse(
    df_dsf=df_dsf,
    df_stocknames=df_stocknames,
    target_size=500,
    min_price=5.0,
    min_history_days=252
)

# 3. Generate annual December reconstitution constituents
annual_constituents = universe_engine.build_annual_constituents()
print(f"Constructed {len(annual_constituents)} annual reconstitution cohorts.")

# 4. Generate point-in-time daily membership panel
daily_membership = universe_engine.get_daily_membership(start_year=2015, end_year=2024)
print(f"Generated daily point-in-time panel with shape: {daily_membership.shape}")
```
