#!/usr/bin/env python3
import os
import sys
import urllib.request
import zipfile

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
        print("Veuillez télécharger le dataset Cityscapes depuis https://cityscapes-dataset.com et utiliser le script ./datasets/prepare_cityscapes_dataset.py.")
        print("Vous devez télécharger gtFine_trainvaltest.zip et leftImg8bit_trainvaltest.zip. Pour plus d'instructions, veuillez consulter ./datasets/prepare_cityscapes_dataset.py")
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
