from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import sys
import urllib.request
import zipfile

def get_dataloader(dataset_name="horse2zebra", image_size=256, batch_size=1, num_workers=4):
    """
    Charge un DataLoader pour un dataset spécifique situé dans le dossier 'datasets/'.
    Par défaut, le dataset 'horse2zebra' est utilisé.

    Parameters
    ----------
    dataset_name : str, optional
        Nom du sous-dossier dans 'datasets/' (par défaut : 'horse2zebra').
    image_size : int, optional
        Taille (carrée) à laquelle les images sont redimensionnées (par défaut : 256).
    batch_size : int, optional
        Taille des lots pour l'entraînement (par défaut : 1).
    num_workers : int, optional
        Nombre de workers utilisés pour le chargement parallèle (par défaut : 4).

    Returns
    -------
    DataLoader
        Un itérateur sur des paires (image_A, image_B) pour l'entraînement.
    """

    # Chemins vers les dossiers A (source) et B (cible)
    path = os.path.join("datasets", dataset_name)
    dir_A = os.path.join(path, "trainA")
    dir_B = os.path.join(path, "trainB")

    # Vérification de l’existence des dossiers attendus
    if not os.path.exists(dir_A) or not os.path.exists(dir_B):
        raise FileNotFoundError(f"Les dossiers '{dir_A}' ou '{dir_B}' n'existent pas.")

    # Transformation standardisée des images : redimensionnement, recadrage aléatoire, normalisation
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Chargement des deux domaines avec ImageFolder
    dataset_A = datasets.ImageFolder(root=os.path.join(path, "trainA"), transform=transform)
    dataset_B = datasets.ImageFolder(root=os.path.join(path, "trainB"), transform=transform)

    # On suppose que les deux datasets ont la même taille et sont alignés
    class PairedDataset(Dataset):
        """
        Dataset personnalisé pour charger des paires d'images à partir de deux ensembles distincts (domaine A et domaine B).

        Chaque appel retourne une paire (image_A, image_B) correspondant à un même index, tronqué à la taille minimale des deux ensembles.

        Parameters
        ----------
        dataset_A : torchvision.datasets.ImageFolder
            Ensemble de données pour le domaine A.
        dataset_B : torchvision.datasets.ImageFolder
            Ensemble de données pour le domaine B.

        Attributes
        ----------
        dataset_A : Dataset
            Données du domaine A.
        dataset_B : Dataset
            Données du domaine B.
        length : int
            Longueur minimale entre dataset_A et dataset_B.

        Methods
        -------
        __len__():
            Retourne la taille du dataset (longueur minimale entre A et B).
        __getitem__(index):
            Retourne la paire d'images à l'index donné.

        Examples
        --------
        >>> dataset_A = datasets.ImageFolder(root='path/to/trainA', transform=transform)
        >>> dataset_B = datasets.ImageFolder(root='path/to/trainB', transform=transform)
        >>> paired_dataset = PairedDataset(dataset_A, dataset_B)
        >>> loader = torch.utils.data.DataLoader(paired_dataset, batch_size=32, shuffle=True)
        """

        def __init__(self, dataset_A, dataset_B):
            """
            Initialise la classe en prenant deux ensembles de données `dataset_A` et `dataset_B` en entrée.

            Parameters
            ----------
            dataset_A : torch.utils.data.Dataset
                Ensemble de données pour le domaine A.
            dataset_B : torch.utils.data.Dataset
                Ensemble de données pour le domaine B.
            """
            self.dataset_A = dataset_A
            self.dataset_B = dataset_B

        def __len__(self):
            """
            Retourne le nombre d'échantillons dans le dataset.

            Returns
            -------
            int
                Longueur minimale entre les ensembles A et B.
            """
            return min(len(self.dataset_A), len(self.dataset_B))  # Assure un alignement minimal


        def __getitem__(self, index):
            """
            Retourne une paire d'images (image_A, image_B) correspondant à l'index spécifié.

            L'index est utilisé pour récupérer les images correspondantes de chaque domaine (A et B).
            L'image de chaque domaine est récupérée par l'index modulo la longueur de l'ensemble, pour permettre
            une correspondance cyclique si les ensembles ont des tailles différentes.

            Parameters
            ----------
            index : int
                Index de l'élément à récupérer.

            Returns
            -------
            tuple of PIL.Image or Tensor
                Une paire d'images correspondant à l'index donné, sous la forme (image_A, image_B).
            """

            img_A, _ = self.dataset_A[index % len(self.dataset_A)]  # Utilise l'index modulo pour assurer un alignement
            img_B, _ = self.dataset_B[index % len(self.dataset_B)]  # Utilise l'index modulo pour assurer un alignement
            return img_A, img_B


    paired_dataset = PairedDataset(dataset_A, dataset_B)

    return DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

def main():
    allowed_datasets = [
        "ae_photos", "apple2orange", "summer2winter_yosemite", "horse2zebra",
        "monet2photo", "cezanne2photo", "ukiyoe2photo", "vangogh2photo", "maps",
        "cityscapes", "facades", "iphone2dslr_flower", "mini", "mini_pix2pix", "mini_colorization"
    ]

    print("Datasets disponibles :")
    print(", ".join(allowed_datasets))
    print()

    # Demande à l'utilisateur de saisir le dataset
    dataset = input("Veuillez entrer le nom du dataset souhaité : ").strip()

    if dataset not in allowed_datasets:
        print("Le dataset '{}' n'est pas reconnu.".format(dataset))
        sys.exit(1)

    if dataset == "cityscapes":
        print("Pour des raisons de licence, nous ne pouvons pas fournir le dataset Cityscapes depuis notre dépôt.")
        print("Veuillez télécharger le dataset Cityscapes depuis https://cityscapes-dataset.com et utiliser le script ./datasets/prepare_cityscapes_dataset.py du git original.")
        print("Vous devez télécharger gtFine_trainvaltest.zip et leftImg8bit_trainvaltest.zip. Pour plus d'instructions, veuillez consulter ./datasets/prepare_cityscapes_dataset.py du git original cf [2]")
        sys.exit(1)

    print("Dataset spécifié [{}]".format(dataset))

    # Construction des chemins et de l'URL
    url = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/{}.zip".format(dataset)
    datasets_dir = os.path.join(".", "datasets")
    zip_file_path = os.path.join(datasets_dir, "{}.zip".format(dataset))
    target_dir = os.path.join(datasets_dir, dataset)

    # S'assurer que le dossier datasets existe
    os.makedirs(datasets_dir, exist_ok=True)

    # Télécharger le fichier zip (similaire à wget -N ; ici, on écrase directement le fichier existant)
    try:
        print("Téléchargement du fichier depuis {}".format(url))
        urllib.request.urlretrieve(url, zip_file_path)
    except Exception as e:
        print("Erreur lors du téléchargement : {}".format(e))
        sys.exit(1)

    # Créer le répertoire cible s'il n'existe pas
    os.makedirs(target_dir, exist_ok=True)

    # Décompresser l'archive
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
        print("Extraction de l'archive terminée")
    except zipfile.BadZipFile as e:
        print("Erreur lors de l'extraction de l'archive : {}".format(e))
        sys.exit(1)

    # Supprimer le fichier zip téléchargé
    try:
        os.remove(zip_file_path)
        print("Fichier zip supprimé")
    except Exception as e:
        print("Erreur lors de la suppression du fichier zip : {}".format(e))
        # On ne quitte pas en cas d'erreur de suppression

if __name__ == "__main__":
    main()
