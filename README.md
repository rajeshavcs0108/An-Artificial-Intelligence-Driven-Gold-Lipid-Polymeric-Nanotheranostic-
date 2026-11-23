# AI-GLPN: TCGA-UVM molecular stratification analysis

Code for the Results and Discussion section of "An Artificial Intelligence-Driven Gold-Lipid-Polymeric Nanotheranostic
Framework for Precision Diagnosis and Vision-Preserving Treatment of Conjunctival Melanoma".

The analysis uses the public TCGA-UVM cohort (80 primary tumours) from the NCI Genomic Data Commons
(https://portal.gdc.cancer.gov/projects/TCGA-UVM). Files are fetched from the cBioPortal datahub copy of the cohort
(`uvm_tcga`, plus `uvm_tcga_pan_can_atlas_2018` for the aneuploidy score). No data are stored in this repository.

## Scope

- TCGA-UVM is a uveal melanoma cohort; conjunctival melanoma is a different tumour, so conclusions about it are extrapolations.
- The dataset has no nanoparticle measurements. The code covers molecular stratification only.
- The published four-cluster labels (15/23/22/20) are not in the downloaded files; clusters here are k-means on expression (k = 4) and are descriptive.

## Run

```bash
pip install -r requirements.txt
python scripts/download_data.py      # downloads ~25 MB into data_uvm/ and data_uvm_pan/
python scripts/uvm_analysis.py       # writes out/uvm_results.json, uvm_cohort.csv, uvm_de.csv, ...
python scripts/uvm_figs.py           # writes figs_uvm/fig1_cohort.png ... fig9_targets.png (300 DPI)
```

Random seed is 42 (`scripts/style_config.py`). Python 3.10+ is recommended.

## Methods in brief

| Step | Method |
|---|---|
| Chromosome 3 status | length-weighted mean log2 copy ratio of chr3 from the segment file; monosomy 3 if <= -0.2 |
| Mutations | non-silent MAF classes; Fisher exact test vs chr3 status, Benjamini-Hochberg q |
| Copy number | 1-Mb bins (gain > 0.2, loss < -0.2); arm-level call at mean log2 ratio > 0.1 / < -0.1 with approximate centromere positions |
| Expression | log2(RSEM + 1), genes with mean >= 1.0 retained (16,087 of 20,513); PCA on 2,000 most variable genes; k-means (k = 2-6) with silhouette |
| Differential expression | Welch t-test, BH q, |log2FC| > 1 |
| Survival | Kaplan-Meier, log-rank, Cox proportional hazards with age (lifelines) |
| Classifier | L2 logistic regression, 5x5 repeated stratified CV, top-50 genes selected inside each training fold |
| Candidate targets | 20 pre-selected genes, Mann-Whitney U, BH q |

## Known caveats

- GNA11 (36/80) and BAP1 (26/80) counts from the downloaded MAF differ from some published values; check variant filtering before reuse.
- Overall survival data exist for 68 patients with 11 deaths, so hazard ratios are imprecise.
- Expression-based classification of chromosome 3 status reaches AUROC 1.000 in internal CV; this is not an external validation.

## Data citation

TCGA-UVM, NCI Genomic Data Commons. cBioPortal datahub: https://github.com/cBioPortal/datahub
[ADD FORMAL DATASET AND PUBLICATION REFERENCES]
