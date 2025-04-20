import torch
import argparse
from .models import Generator, Discriminator  # TODO Importing Generator and Discriminator models
from .data import get_dataloader          # Importing function to get a DataLoader with image pairs
from .utils import save_sample, save_model   # Importing utility functions to save samples and model checkpoints

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
    # Optimiseur Adam pour les générateurs avec betas standards du papier CycleGAN

    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))  # Optimiseur pour D_A
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))  # Optimiseur pour D_B

    # === Fonctions de perte ===
    criterion_GAN = torch.nn.BCELoss()  # Perte GAN classique (binary cross-entropy)
    criterion_cycle = torch.nn.L1Loss()  # Perte de cycle-consistency (différence absolue entre image originale et reconstruite)

    # === Chargement du jeu de données ===
    dataloader = get_dataloader()  # Charger le DataLoader contenant les paires d'images A et B

    # === Boucle d'entraînement principale ===
    for epoch in range(n_epochs):
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)  # Déplacer les images réelles A sur l'appareil (GPU/CPU)
            real_B = real_B.to(device)  # Déplacer les images réelles B sur l'appareil

            # === Entraînement des générateurs ===
            opt_G.zero_grad()  # Réinitialiser les gradients des générateurs avant la rétropropagation

            # Passer les images réelles à travers les générateurs
            fake_B = G_AB(real_A)  # Générer une image fake_B à partir de real_A avec G_AB
            fake_A = G_BA(real_B)  # Générer une image fake_A à partir de real_B avec G_BA

            # Consistance cyclique : récupérer les images originales à partir des images générées
            recov_A = G_BA(fake_B)  # Reconstruire une image proche de real_A à partir de fake_B
            recov_B = G_AB(fake_A)  # Reconstruire une image proche de real_B à partir de fake_A

            # === Calcul de la perte pour les générateurs ===
            # D_B essaie de classer fake_B comme réelle ou fausse
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake_B, real_labels(pred_fake_B.size()))  # Perte GAN pour G_AB

            # D_A essaie de classer fake_A comme réelle ou fausse
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake_A, real_labels(pred_fake_A.size()))  # Perte GAN pour G_BA

            # Perte de cycle-consistency : comparer les images reconstruites avec les images originales
            loss_cycle_A = criterion_cycle(recov_A, real_A)  # Comparer recov_A avec real_A
            loss_cycle_B = criterion_cycle(recov_B, real_B)  # Comparer recov_B avec real_B

            # Calcul de la perte totale pour les générateurs (adversariale + cycle-consistency)
            loss_G = loss_GAN_AB + loss_GAN_BA + lambda_cycle * (loss_cycle_A + loss_cycle_B)

            # Rétropropagation de la perte totale des générateurs
            loss_G.backward()

            # Mise à jour des paramètres des générateurs
            opt_G.step()


            # === Entraînement des discriminateurs ===
            opt_D_A.zero_grad()  # Réinitialiser les gradients de D_A

            # D_A essaie de classer real_A comme réel
            pred_real_A = D_A(real_A)
            loss_D_A_real = criterion_GAN(pred_real_A, real_labels(pred_real_A.size()))  # Perte pour real_A

            # D_A essaie de classer fake_A comme faux
            pred_fake_A = D_A(fake_A.detach())  # .detach() empêche le calcul des gradients pour G_BA
            loss_D_A_fake = criterion_GAN(pred_fake_A, fake_labels(pred_fake_A.size()))  # Perte pour fake_A

            # Calcul de la perte totale pour D_A (moyenne des pertes pour real_A et fake_A)
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)

            # Rétropropagation de la perte de D_A
            loss_D_A.backward()

            # Mise à jour des paramètres de D_A
            opt_D_A.step()

            # Réinitialisation des gradients de D_B
            opt_D_B.zero_grad()

            # D_B essaie de classer real_B comme réel
            pred_real_B = D_B(real_B)
            loss_D_B_real = criterion_GAN(pred_real_B, real_labels(pred_real_B.size()))  # Perte pour real_B

            # D_B essaie de classer fake_B comme faux
            pred_fake_B = D_B(fake_B.detach())  # .detach() empêche le calcul des gradients pour G_AB
            loss_D_B_fake = criterion_GAN(pred_fake_B, fake_labels(pred_fake_B.size()))  # Perte pour fake_B

            # Calcul de la perte totale pour D_B (moyenne des pertes pour real_B et fake_B)
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)

            # Rétropropagation de la perte de D_B
            loss_D_B.backward()

            # Mise à jour des paramètres de D_B
            opt_D_B.step()

            # Affichage des pertes toutes les 100 itérations
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}] "
                    f"[D_A loss: {loss_D_A.item():.4f}] [D_B loss: {loss_D_B.item():.4f}] "
                    f"[G loss: {loss_G.item():.4f}]")

        # Sauvegarde des modèles et des exemples à la fin de chaque époque
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

    # Parsing des arguments
    args = parser.parse_args()

    # Initialisation de l'appareil selon l'argument
    device = torch.device(args.device)

    # Lancement de l'entraînement avec les paramètres spécifiés
    train_cycle_gan(
        n_epochs=args.epochs,
        lambda_cycle=args.lambda_cycle,
        lr=args.lr,
        device=device,
        save=args.save
    )
