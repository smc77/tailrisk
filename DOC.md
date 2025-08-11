
# Drawdown Risk: Measurement, Prediction, and Mitigation — Project Guide

This is a Quarto **book** project with simulations, empirical analysis, early-warning signals, and risk-mitigation overlays (stop-loss + hedging).

## 1) Prerequisites

- **Quarto**: https://quarto.org/docs/get-started/
- **Python 3.9+** and `pip install -r requirements.txt`

## 2) Render the Book
```bash
quarto render
```
Open `_output/index.html` after rendering.

## 3) Notes
- Empirical sections use **yfinance** (internet required). For offline, place CSVs in `data/` and modify `scripts/empirical_data.py`.
- Figures and tables export to `/figures` during render.
