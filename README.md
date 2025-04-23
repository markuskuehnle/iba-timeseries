# MLP Forecasting

## 1. Notebooks
1. **01_data_validation.ipynb** – inspect & clean raw data  
2. **02_global_ts_model.ipynb**   – build & evaluate global monthly forecast with xgb
3. **03_region_ts_model.ipynb**   – build & evaluate per-region monthly forecasts 
4. **04_prophet_model.ipynb**     – build & evaluate global monthly forecast with prohet
5. **05_sarimax_model.ipynb**     - build & evaluate global seasonal arima forecast
---

## 2. Data Dictionary

| Spalte                               | Erläuterung                                                                                         | Typ         |
|--------------------------------------|-----------------------------------------------------------------------------------------------------|-------------|
| **Bereich**                          | Vertriebsbereich                                                                                    | Categorical |
| **Region**                           | Vertriebsregion (identisch mit Bereich außer A)                                                     | Categorical |
| **MLP-Standort**                     | Numerischer Standort-Schlüssel                                                                      | Categorical |
| **Geschäftsstelle**                  | Numerischer Geschäftsstellen-Schlüssel                                                               | Categorical |
| **GS ist HT**                        | 1=Hochschulteam, 0=Geschäftsstelle                                                                  | Binary      |
| **Berater**                          | Berater-ID                                                                                          | ID          |
| **Eintrittsdatum**                   | Eintrittsdatum des Beraters                                                                         | Date        |
| **Austritt MLP**                     | Kündigungsdatum (ex-post bekannt)                                                                   | Date        |
| **akt. Erlöskategorie**              | Erlöskategorie (0, A, B, E, nicht zugeordnet)                                                       | Categorical |
| **Mitarbeiterkreis**                 | 90=Berater, 91=GL, 99=Geschäftsstelle                                                               | Numerical   |
| **Berater ist GL/LHT**               | 1=GL/LHT, 0=Berater                                                                                 | Binary      |
| **Kalendertag**                      | Stichtag für Kennzahlen                                                                             | Date        |
| **Monat**                            | Monat des Stichtags                                                                                 | Integer     |
| **BISBAKTTG GS Tage**                | Tage im Stichtagsjahr in GS/HT                                                                       | Integer     |
| **CKF_PMM_BIANZB Berateranzahl**     | 1=aktiv zum Stichtag                                                                                 | Binary      |
| **BIANZFAMK Familien**               | Anzahl Familienkunden zum Stichtag                                                                  | Integer     |
| **Anz. NeuFamilien**                 | Neugewinnung Familien YTD                                                                           | Integer     |
| **Anz. Neukunden**                   | Neugewinnung Einzelkunden YTD                                                                        | Integer     |
| **SpB AV … SpB FIN**                 | Familien pro Sparte (AV, KV, SV, VEM, BA, FIN)                                                       | Integer     |
| **BISNAV AV St Neug**                | Stücke Neugeschäft AV YTD                                                                            | Integer     |
| **BIWNAV AV Neug Wesu** (Target)     | Wertungssumme Neugeschäft AV YTD                                                                     | Integer     |
| **dav. BR … dav. PV**                | AV-Schichten 1–3 YTD                                                                                  | Integer     |
| **CKF_PMM_BIZGGAEIN Geld NeuBestg**   | Vermögens-Einzahlungen YTD                                                                           | Numeric     |
| **ImmoVerm.**                        | Stücke vermittelte Immobilien YTD                                                                    | Integer     |
| **Immobilien Volumen**               | Volumen vermittelte Immobilien YTD                                                                   | Numeric     |
| **P AV neu … P Bst o AV**             | Provision YTD (Neugeschäft/Bestand, mit/ohne AV)                                                     | Numeric     |
| **Geschlecht**                       | „männlich“, „weiblich“, –1=GS/HT                                                                     | Categorical |
| **Geburtsjahr**                      | Jahr der Geburt                                                                                      | Integer     |
| **Kategorie**, **Zielgruppe**        | Historische Kategorien (A, B, E, nicht zugeordnet)                                                   | Categorical |
| **Kontrolle Berater/Tag**            | –                                                                         | –           |

---

## 3. Key Findings (Data Validation)
- **Mixed series types**: resets (e.g. `BISNAV AV St Neug`) vs. accumulations (e.g. `BIANZFAMK Familien`).  
- **Pre-2021 artifacts**: non-zero `Immobilien Volumen` before 2021 – consider cutoff.  
- **Data gaps**: many Berater show missing/zero target even when “active.”  
- **Seasonality**: clear at aggregate; individual Berater very noisy.

---

## 4. Modeling Considerations

| Type               | Examples                           | Treatment Recommendation                                   |
|--------------------|------------------------------------|------------------------------------------------------------|
| **Reset yearly**   | `BISNAV AV St Neug`, `BIWNAV AV Neug` | Add `Year` covariate or re-index per business year          |
| **Cumulative**     | `BIANZFAMK Familien`, `ImmoVerm.`   | Use as-is or model first differences                       |
| **Flows (monthly)**| `Anz. Neukunden`, `P AV neu`        | Use as-is; consider smoothing                              |

- **Global/Regional**: aggregate monthly, add cyclical features, use forward-CV + hold-out.  
- **Berater-level**: high variance/sparsity → use grouping or hierarchical models; often global TS suffices.

---

## 5. Shift-Based Training Logic
We train on features at time *t* and target at *t+1*:
1. Build X at period *t* (lags, `sin`/`cos(month)` etc.)  
2. Target = series shifted by one month  
3. Model learns X^(t) → y^(t+1)  
4. For multi-step: iterate, feeding predictions back if target lags are used.

---

## 6. Recommendations
- Prefer **global TS model** for stability.  
- Consider advanced features (holidays, business-year) or hierarchical stacking for further accuracy.

**Model Comparison**

| Model                      | MAE (last 6 mo)    | MAPE (last 6 mo) |
|----------------------------|--------------------|-----------------:|
| **Global TS (XGB)**     | 138 446 276        | 10.21 %         |
| **Regional TS**            |  **21 595 636**        |  7.92 %         |
| **Global Prophet**         | 108 688 733        |  7.87 %      |
| **Global SARIMAX**         | 46 731 839       |  **3.39 %**       |


- The **regional model** still wins on absolute error (MAE) and delivers sub-8 % MAPE.  
- **SARIMAX** (global): Its 3.4 % MAPE is hard to beat, it’s easy to retrain, and it directly models the 12-month seasonality 

---

## 7. Setup


### 7.1 With uv (cross-platform)
https://github.com/astral-sh/uv

```bash
# Install uv (once)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell):
irm https://astral.sh/uv/install.ps1 | iex

# Initialize project and pin Python:
uv init
uv python pin    # reads .python-version

# Create venv and install deps:
uv venv
uv add pandas numpy scikit-learn xgboost joblib optuna plotly statsmodels seaborn matplotlib openpyxl
uv sync
```

### 7.2 With pip + venv
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 7.3 Add Data to data folder

data/4_PrognoseCase_AV_NG_-_Daten