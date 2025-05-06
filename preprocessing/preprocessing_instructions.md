## Preprocessing instructions

Steps for loading dataset:

1. Install Git LFS:

    Using Homebrew (macOS):

        If you have Homebrew installed, you can install Git LFS by running:
            brew install git-lfs

    Using Package Manager (Linux):

        For Debian-based systems (like Ubuntu), you can use:
            sudo apt-get install git-lfs

        For Red Hat-based systems (like Fedora), you can use:
            sudo dnf install git-lfs

    Using Installer (Windows):

        Download the Git LFS installer from the Git LFS website and run it.

2. Initialize Git LFS:

    After installing Git LFS, you need to initialize it in your repository:

        git lfs install
    
    You can verify the initilization with:

        git lfs version

3. Load dataset:

    Load the dataset with:
    
        git clone https://huggingface.co/datasets/CDuong04/NGAFID-LOCI-Data

Preprocess the dataset using the script in preprocessing.py.

Example usage:
```
python preprocessing/preprocessing.py \
    NGAFID-LOCI-Data/flights \
    -o preprocessed_data \
    -pad 10000 \
    -drop 1000 \
    -na zero \
    -cols preprocessing/default_columns.txt \
    --split
```

Note: you have to run the file in a parent directory of the input data and the preprocessing script. Argument paths should all be relative to this directory.

Total unique columns: 121 - 

- AOASimple 
- AltAGL 
- AltB 
- AltGPS 
- AltInd 
- AltMSL 
- AltMSL Lag Diff 
- BaroA 
- CAS 
- COM1 
- COM2 
- CRS 
- Coordination Index 
- DensityRatio 
- E1 CHT Divergence 
- E1 CHT1 
- E1 CHT2 
- E1 CHT3 
- E1 CHT4 
- E1 EGT Divergence 
- E1 EGT1 
- E1 EGT2 
- E1 EGT3 
- E1 EGT4 
- E1 FFlow 
- E1 MAP 
- E1 OilP 
- E1 OilT 
- E1 RPM 
- E2 CHT1 
- E2 EGT Divergence 
- E2 EGT1 
- E2 EGT2 
- E2 EGT3 
- E2 EGT4 
- E2 FFlow 
- E2 MAP 
- E2 OilP 
- E2 OilT 
- E2 RPM 
- FQtyL 
- FQtyR 
- GndSpd 
- HAL 
- HCDI 
- HDG 
- HPLfd 
- HPLwas 
- IAS 
- LOC
-I Index 
- LatAc 
- MagVar 
- NormAc 
- OAT 
- PichC 
- Pitch 
- Roll 
- RollC 
- Stall Index 
- TAS 
- TRK 
- Total Fuel 
- True Airspeed(ft/min) 
- VAL 
- VCDI 
- VPLwas 
- VSpd 
- VSpd Calculated 
- VSpdG 
- WndDr 
- WndSpd 
- amp1 
- amp2 
- volt1 
- volt2