import os
import torch
from torchvision.utils import save_image

# === Fonction pour sauvegarder des exemples d'images générées ===
def save_sample(G_AB, G_BA, real_A, real_B, epoch, save_dir="samples"):
    """
    Sauvegarde des exemples d'images générées par les générateurs pour visualisation.

    Parameters
    ----------
    G_AB : torch.nn.Module
        Générateur transformant les images du domaine A vers B.
    G_BA : torch.nn.Module
        Générateur transformant les images du domaine B vers A.
    real_A : torch.Tensor
        Batch d'images réelles du domaine A.
    real_B : torch.Tensor
        Batch d'images réelles du domaine B.
    epoch : int
        Numéro de l'époque actuelle (utilisé pour nommer les fichiers).
    save_dir : str, optional
        Répertoire où sauvegarder les images générées (par défaut : "samples").

    Returns
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)

    # Génération des images synthétiques
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)

    # Sauvegarde des images réelles et générées
    save_image(real_A, os.path.join(save_dir, f"real_A_epoch_{epoch}.png"), nrow=8, normalize=True)
    save_image(fake_B, os.path.join(save_dir, f"fake_B_epoch_{epoch}.png"), nrow=8, normalize=True)
    save_image(real_B, os.path.join(save_dir, f"real_B_epoch_{epoch}.png"), nrow=8, normalize=True)
    save_image(fake_A, os.path.join(save_dir, f"fake_A_epoch_{epoch}.png"), nrow=8, normalize=True)

    print(f"Samples saved for epoch {epoch} in {save_dir}")

# === Fonction pour sauvegarder les modèles ===
def save_model(model, epoch, save_dir="models"):
    """
    Sauvegarde le modèle à un état donné dans un répertoire spécifique.

    Parameters
    ----------
    model : torch.nn.Module
        Le modèle à sauvegarder (par exemple, le générateur ou le discriminateur).
    epoch : int
        L'époque actuelle (utilisée pour nommer le fichier de sauvegarde).
    save_dir : str, optional
        Le répertoire où sauvegarder le modèle (par défaut : "models").

    Returns
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarde des poids des modèles
    torch.save(G_AB.state_dict(), os.path.join(save_dir, f"G_AB_epoch_{epoch}.pth"))
    torch.save(G_BA.state_dict(), os.path.join(save_dir, f"G_BA_epoch_{epoch}.pth"))
    torch.save(D_A.state_dict(), os.path.join(save_dir, f"D_A_epoch_{epoch}.pth"))
    torch.save(D_B.state_dict(), os.path.join(save_dir, f"D_B_epoch_{epoch}.pth"))

    print(f"Models saved for epoch {epoch} in {save_dir}")
