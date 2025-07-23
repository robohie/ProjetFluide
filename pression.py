import numpy as np
import matplotlib.pyplot as plt
import sympy
from fluidflow import ProfilJoukowski  # ou un autre objet compatible


def plot_pressure_zones(objet, x_lim=(-5, 5), y_lim=(-5, 5), density=30):
    """
    Trace les zones qualitatives de pression (haute/basse) à partir du champ de vitesse.
    Dépend des objets de fluidflow (ex: ProfilJoukowski).
    """
    # Grille comme dans fluidflow.champ
    X, Y = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], density),
        np.linspace(y_lim[0], y_lim[1], density)
    )

    dphi_x_func = sympy.lambdify((objet.x, objet.y), objet.d_phi_x, "numpy")
    dphi_y_func = sympy.lambdify((objet.x, objet.y), objet.d_phi_y, "numpy")

    U = dphi_x_func(X, Y)
    V = dphi_y_func(X, Y)

    speed = np.sqrt(U ** 2 + V ** 2)
    norm_speed = (speed - np.min(speed)) / (np.max(speed) - np.min(speed))
    qualitative_pressure = 1 - norm_speed  # Haute vitesse = basse pression

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, qualitative_pressure, cmap='coolwarm', levels=50)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Zones qualitatives de pression\n(Rouge : haute pression, Bleu : basse pression)")
    plt.colorbar(label='Pression relative (qualitative)')
    plt.axis('equal')
    plt.show()


# Exemple d’utilisation dans un script principal :
if __name__ == "__main__":
    # Paramètres comme dans le main de fluidflow.py
    profil = ProfilJoukowski(
        u_inf=1.0,
        alpha=np.radians(5),
        c=1.0,
        x0=-0.1,
        y0=0.1
    )
    plot_pressure_zones(profil)
