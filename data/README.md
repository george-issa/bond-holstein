# Data directory

This directory holds two kinds of files used by the analysis scripts.

## In this Git repository

* `accuracies/accs_w<w>_a<alpha>_L<L>_seed<seed>-<sID>.csv` — LBC test
  accuracy curves (one accuracy per guessed critical inverse temperature)
  produced by `scripts/lbc_train.py` for each `(alpha, seed, sID)` combination.
* `published_phase_diagram.csv` — reference `(method, 1/lambda, T)` values
  used by `scripts/figure7_phase_diagram.py` to draw Fig. 7.

## Distributed via Zenodo (not GitHub)

The bulk DQMC-generated electron-density snapshot CSVs live in directories
of the form

```
electron_densities_w<w>_a<alpha>_L<L>_csv-<sID>/      (sweep over beta at fixed alpha)
electron_densities_w<w>_b<beta>_L<L>_csv-<sID>/       (sweep over 1/lambda at fixed beta)
```

Each combined CSV inside such a directory has shape `(N_samples, L*L + 1)`:
the first `L*L` columns are one electron-density snapshot, the last column is
the sweep parameter (`beta` or `1/lambda`).

These files total roughly 1.8 GB and exceed GitHub's per-file size cap, so
they are deposited on Zenodo:

> **Zenodo record:** _DOI to be assigned upon publication._

Download the archive from Zenodo and unpack it into `data/`, recreating the
`electron_densities_*` subdirectories, before running the figure scripts that
need raw snapshots (i.e. PCA / t-SNE / new LBC training runs).
