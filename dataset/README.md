# Raw data

The notebooks read only the files below. Run `python dataset/download_data.py` from the repository root to fetch the
two files that are not stored here and to verify all four checksums.

| Panel | File | Source | Licence / terms | Stored here |
|---|---|---|---|---|
| Rossmann Store Sales (daily store revenue, 1115 stores, 2013-01-01 to 2015-07-31) | `raw/rossmann/train.csv`, `raw/rossmann/store.csv` | [Kaggle competition](https://www.kaggle.com/competitions/rossmann-store-sales/data) | Kaggle competition rules (no redistribution) | no, downloaded with the Kaggle API |
| UCI Sales Transactions Weekly (811 products × 52 weeks) | `raw/UCI_SalesTransactionsWeekly/source.zip` | [UCI Machine Learning Repository](https://doi.org/10.24432/C5XS4Q) | CC BY 4.0 | yes |
| M5 Forecasting – Accuracy (30,490 Walmart item-store series, 1941 days) | `raw/m5/m5-forecasting-accuracy.zip` | [Zenodo copy](https://doi.org/10.5281/zenodo.12636070) of the [Kaggle competition data](https://www.kaggle.com/competitions/m5-forecasting-accuracy) | CC BY 4.0 as declared on Zenodo; the original competition terms apply | no, downloaded from Zenodo |

SHA-256 checksums of the files used for the stored notebook outputs:

| File | SHA-256 |
|---|---|
| `raw/rossmann/train.csv` | `f6e4597c142d7d909a13d53b68a8e85c00b9a4c7b5ff40adbb37d6829cc1f4cc` |
| `raw/rossmann/store.csv` | `f56bd124a2849489e6bbb5c000f5fc9640204355e316475c918ae4d089afb344` |
| `raw/UCI_SalesTransactionsWeekly/source.zip` | `dededa20f199af8f2bb96184bb1696befda902849a700bde283fe763afc90ba9` |
| `raw/m5/m5-forecasting-accuracy.zip` | `0349ba38a2efd30d0f5acc6394c1110e140e1a990c650d7b5ca44c5b25dd12f5` |

All intermediate data (merged panels, feature matrices, splits, forecasts) are rebuilt by the notebooks in
`outputs/`, which is not under version control.
