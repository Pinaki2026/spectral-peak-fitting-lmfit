"""
Fitting overlapping peaks in a spectrum with lmfit — companion code for the tutorial.

Builds a synthetic spectrum (two overlapping Lorentzian peaks on a sloped
background, with noise), fits it with lmfit, prints the parameters with
uncertainties, and saves a publication-style figure with residuals.

Run:  python fit_peaks.py
Needs: numpy, matplotlib, lmfit
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel, LinearModel

rng = np.random.default_rng(seed=7)

# ---------------------------------------------------------------- synthetic data
x = np.linspace(600, 720, 600)  # e.g. wavelength in nm


def lorentz(x, amp, cen, sig):
    """Area-normalised Lorentzian (same convention as lmfit)."""
    return amp / np.pi * sig / ((x - cen) ** 2 + sig ** 2)


true = dict(p1=(900, 652.0, 6.0), p2=(500, 668.0, 8.0), slope=0.02, icpt=-8.0)
y_clean = (lorentz(x, *true["p1"]) + lorentz(x, *true["p2"])
           + true["slope"] * x + true["icpt"])
y = y_clean + rng.normal(0, 0.6, x.size)

# ---------------------------------------------------------------- model
peak1 = LorentzianModel(prefix="p1_")
peak2 = LorentzianModel(prefix="p2_")
bkg = LinearModel(prefix="bg_")
model = peak1 + peak2 + bkg

params = model.make_params()
# Sensible starting guesses + physical bounds
params["p1_center"].set(value=650, min=630, max=660)
params["p2_center"].set(value=670, min=660, max=690)
for p in ("p1_", "p2_"):
    params[p + "amplitude"].set(value=500, min=0)
    params[p + "sigma"].set(value=5, min=0.5, max=30)
params["bg_slope"].set(value=0)
params["bg_intercept"].set(value=0)

result = model.fit(y, params, x=x)
print(result.fit_report(min_correl=0.5))

# ---------------------------------------------------------------- figure
comps = result.eval_components(x=x)
fig, (ax, axr) = plt.subplots(2, 1, figsize=(6, 4.6), sharex=True,
                              gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
ax.plot(x, y, ".", ms=2.5, color="0.55", label="data")
ax.plot(x, result.best_fit, "-", lw=1.6, color="k", label="total fit")
ax.fill_between(x, comps["bg_"], comps["p1_"] + comps["bg_"], alpha=0.35,
                color="#1f77b4", label="peak 1")
ax.fill_between(x, comps["bg_"], comps["p2_"] + comps["bg_"], alpha=0.35,
                color="#d62728", label="peak 2")
ax.plot(x, comps["bg_"], "--", lw=1, color="0.3", label="background")
ax.set_ylabel("Intensity (arb. u.)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
for p, c, (tx, ty) in (("p1_", "#1f77b4", (606, 38)), ("p2_", "#d62728", (684, 22))):
    cen = result.params[p + "center"]
    fw = result.params[p + "fwhm"]
    ax.text(tx, ty, f"centre {cen.value:.2f} ± {cen.stderr:.2f} nm\n"
                    f"FWHM {fw.value:.2f} ± {fw.stderr:.2f} nm",
            fontsize=7.5, color=c)
axr.axhline(0, color="0.3", lw=0.8)
axr.plot(x, y - result.best_fit, ".", ms=2, color="0.4")
axr.set_xlabel("Wavelength (nm)")
axr.set_ylabel("Resid.")
for a in (ax, axr):
    a.tick_params(direction="in", top=True, right=True)
fig.align_ylabels()
fig.savefig("peak_fit.png", dpi=200, bbox_inches="tight")
print("saved peak_fit.png")
