# Fitting Overlapping Peaks in a Spectrum with Python and lmfit

*By Pinaki Chatterjee, PhD candidate in Physics (photonics), IISc Bengaluru. Full code: [`fit_peaks.py`](fit_peaks.py) — runs as-is with numpy, matplotlib and lmfit.*

Every optics, chemistry or materials lab eventually runs into the same task. A spectrum shows two peaks that overlap, and you need each one's centre, width and area, with honest error bars. `scipy.optimize.curve_fit` can do it, but you have to write the model yourself and do the parameter bookkeeping by hand. **lmfit** builds the model from named pieces, lets you put bounds on any parameter, and reports uncertainties and correlations automatically.

This tutorial shows you how to:

1. build a composite model (two peaks plus a background) from lmfit's built-in pieces,
2. choose starting values and bounds so the fit converges reliably,
3. read the fit report, including the correlations most people skip,
4. make a figure ready for publication, with the components and a residuals panel.

## 1. The data

We use a synthetic spectrum so that you can check the fit against the true answer. It has two Lorentzian peaks at 652 nm and 668 nm, with half-widths of 6 nm and 8 nm, on a gently sloped background, plus Gaussian noise with σ = 0.6.

```python
x = np.linspace(600, 720, 600)
y_clean = lorentz(x, 900, 652.0, 6.0) + lorentz(x, 500, 668.0, 8.0) + 0.02 * x - 8.0
y = y_clean + rng.normal(0, 0.6, x.size)
```

The peaks are only about 2.3 half-widths apart, so the second one shows up as a shoulder rather than a separate peak. This is exactly the case where eyeballing the peak positions fails.

## 2. Building the model from parts

lmfit models can be added together. Give each piece a **prefix** so that its parameter names don't clash:

```python
from lmfit.models import LorentzianModel, LinearModel

peak1 = LorentzianModel(prefix="p1_")
peak2 = LorentzianModel(prefix="p2_")
bkg   = LinearModel(prefix="bg_")
model = peak1 + peak2 + bkg
params = model.make_params()
```

`params` now holds `p1_amplitude`, `p1_center`, `p1_sigma`, the same three for `p2_`, and `bg_slope` and `bg_intercept`. lmfit also creates **derived** parameters, `p1_fwhm` and `p1_height`, and propagates uncertainties to them for you.

> **Convention check.** In lmfit, `amplitude` is the *area* under the peak and `sigma` is the *half*-width at half-maximum, so FWHM = 2σ. Mixing up these conventions is the most common source of factor-of-2 errors in published linewidths.

## 3. Starting values and bounds

Nonlinear fits are only as good as their starting point. Use what you can see in the plot, then bound the parameters so the two peaks cannot swap places or collapse to zero width:

```python
params["p1_center"].set(value=650, min=630, max=660)
params["p2_center"].set(value=670, min=660, max=690)
for p in ("p1_", "p2_"):
    params[p + "amplitude"].set(value=500, min=0)
    params[p + "sigma"].set(value=5, min=0.5, max=30)
```

Two rules of thumb apply. Separate the allowed centre ranges at a point between the peaks, which stops the optimiser from swapping them. Keep the lower bound on width above the spectrometer's resolution, so the fit cannot shrink a peak to fit a single noisy point.

## 4. Fitting and reading the report

```python
result = model.fit(y, params, x=x)
print(result.fit_report(min_correl=0.5))
```

| Parameter | True | Fitted |
|---|---|---|
| Peak 1 centre (nm) | 652.0 | 652.03 ± 0.02 |
| Peak 1 FWHM (nm) | 12.0 | 12.07 ± 0.06 |
| Peak 2 centre (nm) | 668.0 | 668.03 ± 0.06 |
| Peak 2 FWHM (nm) | 16.0 | 15.88 ± 0.19 |

Three parts of the report deserve attention.

- **Reduced χ².** Here it is about 0.31, not 1. That doesn't mean the fit is too good: we never told lmfit the noise level. Pass `weights=1/sigma_y` (σ_y = 0.6 here) and the reduced χ² comes out at about 0.85, close to 1, as you would expect for a correct model. Only then does it become a real goodness-of-fit test. By default lmfit rescales the error bars so that reduced χ² = 1, so the uncertainties above are still sensible.
- **Correlations.** `C(p1_amplitude, p1_sigma) ≈ +0.88` and `C(p1_amplitude, p2_amplitude) ≈ −0.73`. Overlapping peaks trade area with each other. If you report the two areas as independent numbers, their ratio has a larger uncertainty than either error bar suggests. Use `result.uvars` (or the `uncertainties` package) to propagate the errors properly.
- **Background slope and intercept** are almost perfectly anti-correlated (−0.997). That is harmless here. You can reduce it by fitting against `x - x.mean()`.

## 5. A publication-quality figure

`result.eval_components()` returns each component separately. Fill each peak above the background so that readers can see how the shoulder decomposes, and always show the residuals:

```python
comps = result.eval_components(x=x)
ax.fill_between(x, comps["bg_"], comps["p1_"] + comps["bg_"], alpha=0.35)
ax.fill_between(x, comps["bg_"], comps["p2_"] + comps["bg_"], alpha=0.35)
axr.plot(x, y - result.best_fit, ".")
```

The residuals panel tells you more about fit quality than any single number. Structure in the residuals, such as a wiggle under a peak, means the line shape is wrong. Try `VoigtModel` or `PseudoVoigtModel` when instrument broadening adds a Gaussian part to the line shape.

## 6. Checklist for real data

- Subtract dark counts before fitting. Fit the background instead of subtracting a hand-drawn baseline.
- Convert to energy (eV) before fitting if the physics is linear in energy, which it is for excitons and phonons. A Lorentzian in energy is *not* a Lorentzian in wavelength.
- Pass `weights` from shot noise, which is roughly `1/sqrt(counts)` for photon counting.
- Report centre ± error, FWHM ± error, and the model you used, so that others can reproduce the fit.

*The full script, `fit_peaks.py`, reproduces every number and the figure above.*
