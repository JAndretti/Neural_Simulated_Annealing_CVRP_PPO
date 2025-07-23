import os
import zipfile
from tqdm import tqdm
import urllib.request

DOWNLOAD_DB = False

for folder in ["wandb", "res", "bdd"]:
    os.makedirs(folder, exist_ok=True)

link_bdd = [
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-A.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-B.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-E.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-F.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-M.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-P.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-CMT.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-tai.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-Golden.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-Li.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-X.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-XXL.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-D.zip",
    "http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-XML100.zip",
]


if DOWNLOAD_DB:
    # Download the datasets
    print("Downloading datasets...")
    for url in tqdm(link_bdd, desc="Downloading datasets"):
        filename = os.path.basename(url)
        zip_path = os.path.join("bdd", filename)
        # Download the file
        urllib.request.urlretrieve(url, zip_path)
        # Unzip the file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("bdd")
            # Remove the zip file after extraction
            os.remove(zip_path)
