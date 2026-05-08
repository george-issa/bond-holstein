# MATLAB figure scripts

These scripts reproduce the DQMC and atomic-limit panels of the paper from
the processed observables shipped under `data/`.  Run each script from this
directory (`matlab/`) so that the relative paths resolve; the resulting PDF
or PNG is written to the top-level `figures/` directory.

## Script ↔ paper figure mapping

| Script                  | Paper figure                                       | Output                    |
|-------------------------|----------------------------------------------------|---------------------------|
| `figure1.m`             | Fig. 2 — charge density snapshots vs. $\beta$      | `figures/figure1.png`     |
| `figure2.m`             | Fig. 3 — $S(\pi,\pi)$ and finite-size scaling      | `figures/figure2.pdf`     |
| `figure3.m`             | Fig. 4 — $\mathcal{K}$, $\mathcal{V}$, $\mathcal{D}$ vs. temperature | `figures/figure3.pdf` |
| `make_teq0_figure.m`    | Fig. 5 — atomic-limit ($t=0$) specific heat and Binder ratio | `figures/teq0.pdf` |
| `figure4.m`             | Fig. 7 — bond- vs. site-Holstein phase diagram     | `figures/phase_diagram.png` |
| `load_data.m`           | helper used inside `figure1.m`                     | —                         |
| `figure5.m`             | supporting plot of $d^2\mathcal{V}/dT^2$           | (no saveas)               |
| `figure7.m`             | phase-diagram skeleton (loads data, no annotations) | (no saveas)              |

`figure5.m` and `figure7.m` are kept for completeness; they are the
supporting analyses used while preparing Figs. 4 and 7 and do not produce
the published panels on their own.

## Data files

`data/` contains:

- `charge_density_real_space.csv` — $\langle \hat n_i \hat n_0 \rangle$ on $L=8$, $\lambda_{\rm bond}=0.4$, used by `figure1.m`.
- `site_L{8,10,12}.csv`, `sorted_bond_a0.6325_L{8,10,12}.csv` — site- and bond-Holstein $S(\pi,\pi)$ vs. $\beta$ used by `figure2.m`.
- `hopping_energy.csv`, `holstein_energy.csv`, `double_occ.csv` — temperature dependence of $\mathcal{K}$, $\mathcal{V}$, $\mathcal{D}$ used by `figure3.m` and `figure5.m`.
- `CN{8,12,16}new.dat`, `BinderN{8,12,16}new.dat` — classical Monte Carlo data in the $t=0$ limit ($|U_\mathrm{eff}|=1$) used by `make_teq0_figure.m`.
- `phase_diagram.csv` — bond-Holstein DQMC $T_\mathrm{cdw}$ values $(\alpha, \beta_c, \sigma_{\beta_c})$ used by `figure4.m` and `figure7.m`.
- `CDW_betacs.csv`, `cross-over_lambdacs.csv`, `L{8,10,12}.csv` — additional crossover and structure-factor data used in supporting plots.

Note that `data/phase_diagram.csv` here stores raw $(\alpha,\beta_c)$
points, while `../data/published_phase_diagram.csv` stores the
$(1/\lambda, T)$ values from each method that are plotted in Fig. 7 by
`scripts/figure7_phase_diagram.py`.  Both describe the same phase
boundary; they are kept separate so each plotting workflow can be run
independently.

## Requirements

The smoothing/spline calls (`csaps`, `fnplt`) require the **MATLAB Curve
Fitting Toolbox**.  Tested with MATLAB R2023b; no further toolboxes are
needed.

## Running

```matlab
cd matlab
figure1            % Fig. 2
figure2            % Fig. 3
figure3            % Fig. 4
make_teq0_figure   % Fig. 5
figure4            % Fig. 7
```
