# What's Your Dating-Market Percentile?

### ▶️ Live site: **https://GenEqui.github.io/dating-market-percentile/**

A for-fun web toy that estimates where you rank among singles competing for
attention in your city. **The whole app is a single self-contained `index.html`** —
no backend, no build step, no data leaves the browser. Open the live link above, or
just open the file locally.

> ⚠️ Entertainment only. The weights reflect *population-average* research on dating
> preferences — they say nothing about anyone's worth, and real attraction is
> gloriously unpredictable.

## What's under the hood

- **Individual desirability score** — each attribute (looks, age, income, education,
  occupation, height, personality, lifestyle) becomes a 0–100 sub-score off a
  reference distribution, then combined with weights that reflect the *revealed*
  preferences of the gender you're trying to attract (Hitsch/Hortaçsu/Ariely 2010,
  Bruch & Newman 2018, Buss 1989, US Census percentiles).
- **Market layer (US cities): a real Choo & Siow (2006) marriage-matching model.**
  Types = age-band × socioeconomic-desirability tercile. The gains-to-marriage
  surface `π` is **estimated on ACS 2023 PUMS** (16 states, ~306k married couples),
  and the app re-solves the market-clearing equilibrium under each metro's actual
  age × SES × sex population (counties → PUMAs via the Census crosswalk) to produce
  a market adjustment *with equilibrium spillovers*. International cities use a
  simpler sex-ratio model.

## Files

| File | Role |
|------|------|
| `index.html` | the entire app (self-contained; the estimated `π` + metro supplies are embedded) |
| `choo_siow.py` | Choo–Siow gains estimator (eq. 11) + IPFP equilibrium solver |
| `build_gains2.py` | estimates `π` on ACS PUMS with age×SES types + per-metro supplies |
| `choo_historical.py` | validation: ACS 2010 vs 2023 reproduces the paper's young-adult finding |
| `gains_us2.json` | estimated parameters (also embedded in `index.html`) |

## Run it

Open `index.html` in any browser, or serve the folder:

```bash
python3 -m http.server 8000    # then visit http://localhost:8000
```

To re-estimate from data, download ACS 1-year PUMS person files from the Census
Bureau and point `build_gains2.py` at them.

## Data & sources

ACS 2023 & 2010 1-year PUMS (U.S. Census Bureau, public domain). Choo, E. & Siow, A.
(2006), *Who Marries Whom and Why*, Journal of Political Economy 114(1). The paper
PDF is intentionally not included in this repo.
