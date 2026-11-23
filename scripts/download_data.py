"""Download the TCGA-UVM files used in the study from the cBioPortal datahub (GitHub, Git LFS media endpoint).

Source study: cBioPortal 'uvm_tcga' (TCGA-UVM, 80 primary tumours), derived from the NCI Genomic Data Commons
project TCGA-UVM: https://portal.gdc.cancer.gov/projects/TCGA-UVM
"""
import os, sys, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"
FILES = {
    "data_uvm": ("uvm_tcga", ["data_clinical_patient.txt", "data_clinical_sample.txt", "data_mutations.txt",
                              "data_cna.txt", "data_mrna_seq_v2_rsem.txt", "data_cna_hg19.seg"]),
    "data_uvm_pan": ("uvm_tcga_pan_can_atlas_2018", ["data_clinical_sample.txt"]),   # aneuploidy score
}
for folder, (study, names) in FILES.items():
    os.makedirs(os.path.join(ROOT, folder), exist_ok=True)
    for n in names:
        dest = os.path.join(ROOT, folder, n)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print("exists", dest); continue
        url = f"{BASE}/{study}/{n}"; print("downloading", url)
        urllib.request.urlretrieve(url, dest)
        if os.path.getsize(dest) < 1000:
            sys.exit(f"{n} looks like a Git LFS pointer or an error page; check the URL")
print("done")
