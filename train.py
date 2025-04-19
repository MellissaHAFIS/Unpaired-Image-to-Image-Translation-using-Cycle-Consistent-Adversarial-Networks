import torch
import argparse
from .models import Generator, Discriminator      # À adapter : modules définissant l’architecture des générateurs et discriminateurs
from .dataset import get_dataloader              # Fonction retournant un DataLoader fournissant des couples (image_A, image_B)
from .utils import save_sample, save_model       # Fonctions utilitaires pour sauvegarder des échantillons et checkpoints

# === Fonction pour définir les étiquettes GAN ===
def real_labels(size, smooth=True):
    """
    Génère des étiquettes "réelles" pour l'entraînement du discriminateur, avec option de label smoothing.

    Parameters
    ----------
    size : torch.Size
        Taille du batch des étiquettes à générer.
    smooth : bool, optional
        Si True, applique un label smoothing (0.9 au lieu de 1.0). Par défaut à True.

    Returns
    -------
    torch.Tensor
        Un tenseur de taille `size` rempli de 1.0 (ou 0.9 si `smooth` est True), représentant les étiquettes "réelles".
    """
    return torch.full(size, 0.9 if smooth else 1.0, device=device)

def fake_labels(size):
    """
    Génère des étiquettes "fausses" pour l'entraînement du discriminateur.

    Parameters
    ----------
    size : torch.Size
        Taille du batch des étiquettes à générer.

    Returns
    -------
    torch.Tensor
        Un tenseur de même taille que `size`, rempli de 0.0 pour représenter des étiquettes "fausses".
    """
    return torch.zeros(size, device=device)


def train_cycle_gan(n_epochs, lambda_cycle, lr, device, save=True):
    """
    Entraîne un modèle CycleGAN avec pertes adversariales et cohérence cyclique.

    Parameters
    ----------
    n_epochs : int
        Nombre d'époques d'entraînement.
    lambda_cycle : float
        Poids de la perte cyclique.
    lr : float
        Taux d'apprentissage des optimiseurs.
    device : torch.device
        Appareil utilisé pour l'entraînement (CPU ou GPU).
    save : bool, optional
        Si True, sauvegarde les modèles et exemples à chaque époque (par défaut: True).

    Returns
    -------
    None
    """


    # === Initialisation des modèles ===
    G_AB = Generator().to(device)  # Générateur G_AB : A → B
    G_BA = Generator().to(device)  # Générateur G_BA : B → A
    D_A = Discriminator().to(device)  # Discriminateur D_A pour le domaine A
    D_B = Discriminator().to(device)  # Discriminateur D_B pour le domaine B

    # === Optimiseurs ===
    opt_G = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=lr,
        betas=(0.5, 0.999)
    )
    # Adam optimizer like in the paper with standard choice for betas
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    # === Fonctions de perte ===
    criterion_GAN = torch.nn.BCELoss()  # Perte GAN classique
    criterion_cycle = torch.nn.L1Loss()  # Perte de cycle-consistency (L1)

    # === Chargement du jeu de données ===
    dataloader = get_dataloader()  # Charger les données

    # === Boucle d'entraînement principale ===
    for epoch in range(n_epochs):
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # === Entraînement des générateurs ===
            # Zero gradients before the backward pass
            opt_G.zero_grad()

            # Forward pass through the generators: G_AB generates fake_B from real_A
            fake_B = G_AB(real_A)

            # Forward pass through the other generator: G_BA generates fake_A from real_B
            fake_A = G_BA(real_B)

            # Cycle consistency: recover the original images from the generated ones
            recov_A = G_BA(fake_B)  # G_BA should return an image close to real_A
            recov_B = G_AB(fake_A)  # G_AB should return an image close to real_B

            # Discriminator D_B tries to classify fake_B as real or fake
            pred_fake_B = D_B(fake_B)
            # Calculate GAN loss for G_AB using fake_B and label it as real (real_labels)
            loss_GAN_AB = criterion_GAN(pred_fake_B, real_labels(pred_fake_B.size()))

            # Discriminator D_A tries to classify fake_A as real or fake
            pred_fake_A = D_A(fake_A)
            # Calculate GAN loss for G_BA using fake_A and label it as real (real_labels)
            loss_GAN_BA = criterion_GAN(pred_fake_A, real_labels(pred_fake_A.size()))

            # Cycle consistency loss: the difference between the recovered images and the originals
            loss_cycle_A = criterion_cycle(recov_A, real_A)  # Compare recovered_A to real_A
            loss_cycle_B = criterion_cycle(recov_B, real_B)  # Compare recovered_B to real_B

            # Total generator loss: combines the adversarial losses and cycle consistency losses
            loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_A + loss_cycle_B)

            # Backpropagate the total generator loss to update the generator weights
            loss_G.backward()

            # Update the generator parameters based on the gradients computed during the backward pass
            opt_G.step()


            # === Entraînement des discriminateurs ===
            # Zero gradients before the backward pass for discriminator D_A
            opt_D_A.zero_grad()

            # Discriminator D_A tries to classify real_A as real
            pred_real_A = D_A(real_A)
            # Calculate the loss for D_A when it classifies real_A as real (real_labels)
            loss_D_A_real = criterion_GAN(pred_real_A, real_labels(pred_real_A.size()))

            # Discriminator D_A tries to classify fake_A (generated by G_BA) as real
            pred_fake_A = D_A(fake_A.detach())  # .detach() prevents gradients from flowing through the generator
            # Calculate the loss for D_A when it classifies fake_A as fake (fake_labels)
            loss_D_A_fake = criterion_GAN(pred_fake_A, fake_labels(pred_fake_A.size()))

            # Total loss for D_A: average the loss for real and fake images
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)

            # Backpropagate the total loss for D_A
            loss_D_A.backward()

            # Update the parameters of D_A based on the computed gradients
            opt_D_A.step()

            # Zero gradients before the backward pass for discriminator D_B
            opt_D_B.zero_grad()

            # Discriminator D_B tries to classify real_B as real
            pred_real_B = D_B(real_B)
            # Calculate the loss for D_B when it classifies real_B as real (real_labels)
            loss_D_B_real = criterion_GAN(pred_real_B, real_labels(pred_real_B.size()))

            # Discriminator D_B tries to classify fake_B (generated by G_AB) as real
            pred_fake_B = D_B(fake_B.detach())  # .detach() prevents gradients from flowing through the generator
            # Calculate the loss for D_B when it classifies fake_B as fake (fake_labels)
            loss_D_B_fake = criterion_GAN(pred_fake_B, fake_labels(pred_fake_B.size()))

            # Total loss for D_B: average the loss for real and fake images
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)

            # Backpropagate the total loss for D_B
            loss_D_B.backward()

            # Update the parameters of D_B based on the computed gradients
            opt_D_B.step()

            # Every 100 batches, print the current losses for D_A, D_B, and G
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}] "
                    f"[D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}]")


        # Sauvegarde des modèles et des exemples
        if save :
            save_sample(G_AB, G_BA, real_A, real_B, epoch)
            save_model(G_AB, G_BA, D_A, D_B, epoch)


# === Code principal pour exécuter le script ===
if __name__ == '__main__':
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Entraînement d'un modèle CycleGAN.")
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques d\'entraînement (par défaut: 100)')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Poids de la perte cyclique (par défaut: 10.0)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Taux d\'apprentissage (par défaut: 2e-4)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda'],
                        help='Périphérique d\'entraînement ("cuda" ou "cpu", par défaut: "cuda" si disponible)')
    parser.add_argument('--save', action='store_true', help='Sauvegarde les modèles et exemples après chaque époque.')


    args = parser.parse_args()

    # Initialisation de l'appareil selon l'argument
    device = torch.device(args.device)

    # Lancement de l'entraînement
    train_cycle_gan(
        n_epochs=args.epochs,
        lambda_cycle=args.lambda_cycle,
        lr=args.lr,
        device=device,
        save=args.save
    )

