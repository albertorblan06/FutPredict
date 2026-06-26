import os
import kagglehub

os.environ["KAGGLE_API_TOKEN"] = "KGAT_1c4df5dc4a81d0380d6a72749cef25f6"

def download():
    print("Downloading Match Data...")
    path1 = kagglehub.dataset_download("swaptr/fifa-wc-2026-matches")
    print("Match Data Path:", path1)
    
    print("\nDownloading Player Performance Data...")
    path2 = kagglehub.dataset_download("rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset")
    print("Player Data Path:", path2)

if __name__ == "__main__":
    download()
