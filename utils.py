import os
import torch
from torchvision.utils import save_image
import random

class ImagePool:
    """
    History buffer for generated images.
    Stores up to pool_size images, et renvoie parfois de vieilles images au discriminateur
    pour stabiliser l'entraînement (technique du papier CycleGAN).
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        images: batch Tensor de forme (B, C, H, W)
        Retourne un batch Tensor de même forme, contenant un mélange
        d'images récentes et d'anciennes images du pool.
        """
        if self.pool_size == 0:
            return images

        return_images = []
        for img in images:
            img = img.unsqueeze(0).detach().clone()
            if self.num_imgs < self.pool_size:
                # Remplissage du pool jusqu'à pool_size
                self.images.append(img)
                self.num_imgs += 1
                return_images.append(img)
            else:
                # 50% chance de renvoyer une ancienne image
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    # Remplacer l'ancienne image par la nouvelle
                    self.images[idx] = img
                    return_images.append(tmp)
                else:
                    # Sinon, renvoyer l'image courante
                    return_images.append(img)
        # Concaténer en un batch
        return torch.cat(return_images, 0)


def save_sample(G_AB, G_BA, real_A, real_B, epoch, save_dir="samples"):
    """
    Génère et sauvegarde des images de démonstration pour chaque époque.
    - real_A → fake_B
    - real_B → fake_A
    """
    os.makedirs(save_dir, exist_ok=True)
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)

    save_image(real_A, os.path.join(save_dir, f"real_A_epoch_{epoch}.png"),
               nrow=8, normalize=True)
    save_image(fake_B, os.path.join(save_dir, f"fake_B_epoch_{epoch}.png"),
               nrow=8, normalize=True)
    save_image(real_B, os.path.join(save_dir, f"real_B_epoch_{epoch}.png"),
               nrow=8, normalize=True)
    save_image(fake_A, os.path.join(save_dir, f"fake_A_epoch_{epoch}.png"),
               nrow=8, normalize=True)

    print(f"📷 Samples saved for epoch {epoch} in {save_dir}")


def save_model(G_AB, G_BA, D_A, D_B, epoch, save_dir="models"):
    """
    Sauvegarde les états complets des 4 réseaux à la fin de chaque époque.
    """
    os.makedirs(save_dir, exist_ok=True)
    torch.save(G_AB.state_dict(), os.path.join(save_dir, f"G_AB_epoch_{epoch}.pth"))
    torch.save(G_BA.state_dict(), os.path.join(save_dir, f"G_BA_epoch_{epoch}.pth"))
    torch.save(D_A.state_dict(), os.path.join(save_dir, f"D_A_epoch_{epoch}.pth"))
    torch.save(D_B.state_dict(), os.path.join(save_dir, f"D_B_epoch_{epoch}.pth"))

    print(f"Models saved for epoch {epoch} in {save_dir}")
