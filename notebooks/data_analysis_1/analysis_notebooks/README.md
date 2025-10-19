# Data Analysis Helpers

This folder now hosts interactive notebooks to profile the historical prices in `../stock_data` prior to building a transformer-based forecasting pipeline.

All notebooks expect Python 3.10+ with `pandas`, `numpy`, and Jupyter available:

```bash
pip install pandas numpy notebook
```

## Notebooks

- `dataset_overview.ipynb` – counts tickers, reports coverage windows, and surfaces the longest and most liquid series. Useful for deciding which symbols to model first and what horizon is available.
- `returns_analysis.ipynb` – computes daily log-return statistics (drift, volatility, skew, kurtosis, signal-to-noise, Sharpe-like ratios, and autocorrelation). These metrics guide feature scaling choices and help determine if a simple trend or volatility focus is appropriate.
- `missing_data_report.ipynb` – checks each series for missing business days and quantifies the worst gaps. This informs whether you need imputation, masking, or ticker filtering before batching sequences for a transformer.

Each notebook exposes a configuration cell where you can point to different data roots, adjust reporting limits, and optionally export the per-symbol metrics to CSV. Run them top-to-bottom after tweaking the parameters.

> Tip: Save parameter choices or interesting output by exporting the notebook to HTML or committing executed notebooks to version control (after stripping large outputs).

Run these diagnostics before modelling to choose clean, liquid tickers, set sequence lengths based on coverage, and decide how to normalise inputs for transformer training.
