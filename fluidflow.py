# ================================
# FACULTE POLYTECHNIQUE
# UNIVERSITE DE KINSHASA / UNIKIN
# ================================
# Projet de MECANIQUE DES FLUIDES
# Deuxième licence en science de
# l'ingénieur : GEI, GC, GM
# ================================
# Membres du groupe :
#     BOSOLINDO EDHIENGENE ROGER (GEI)
#     BAMPIRE NGABO DAVID (GEI)
#     ESAFE ISIMO BENJAMIN (GC)
#     KABONGO MUKENDI ODON (GM)
#     MUKENGE KOLM THADDEE (GEI)
#     TSHIMANGA KABANZA CRIS-BOAZ (GEI)
#     DEBORAH (GC)
# ================================

import sympy
import matplotlib.pyplot as plt
import numpy as np


def latex(objet):
    """Génère une section LaTeX résumant l'écoulement."""
    text = (
        f"\\section*{{Ecoulement : {objet.nom}}}\n"
        f"Potentiel complexe \\[{sympy.latex(objet.f)}\\]\n"
        f"Fonction potentielle \\[{sympy.latex(objet.phi())}\\]\n"
        f"Fonction courant \\[{sympy.latex(objet.psi())}\\]\n"
        f"Champ vectoriel \\[{sympy.latex(objet.vect_v())}\\]\n"
    )
    return text


def courant(objet, a, b, niveaux):
    """Fonction qui trace les lignes de courant
    """
    niveaux = sorted(niveaux)
    x_vals = np.linspace(-a, a, 200)
    y_vals = np.linspace(-b, b, 200)
    X, Y = np.meshgrid(x_vals, y_vals)

    psi = sympy.lambdify((objet.x, objet.y), objet.psi(), "numpy")

    Psi = psi(X, Y)

    fig, axe = plt.subplots(figsize=(8, 6))

    axe.contour(X, Y, Psi, levels=niveaux)
    axe.set_title(label=f"lignes de courant ({objet.nom})")
    axe.grid(True)
    axe.set_aspect("equal")
    fig.tight_layout()

    return fig


def potentiel(objet, a, b, niveaux):
    """Fonction qui trace les équipotentiels
    """
    niveaux = sorted(niveaux)
    x_vals = np.linspace(-a, a, 400)
    y_vals = np.linspace(-b, b, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    phi = sympy.lambdify((objet.x, objet.y), objet.phi(), "numpy")

    Phi = phi(X, Y)

    fig, axe = plt.subplots(figsize=(8, 6))

    axe.contour(X, Y, Phi, levels=niveaux)
    axe.set_title(label=f"Equipotentiels ({objet.nom})")
    axe.grid(True)
    axe.set_aspect("equal")
    fig.tight_layout()

    return fig


def champ(objet, x_lim=(-10, 10), y_lim=(-10, 10), density=20):
    """Affiche le champ vectoriel de vitesse."""
    X, Y = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], density),
        np.linspace(y_lim[0], y_lim[1], density)
    )
    #
    dphi_x_func = sympy.lambdify((objet.x, objet.y), objet.d_phi_x)
    dphi_y_func = sympy.lambdify((objet.x, objet.y), objet.d_phi_y)
    U = dphi_x_func(X, Y)
    V = dphi_y_func(X, Y)
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V)
    plt.grid(True)
    plt.title("Champ de vitesse")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


class PlanPlus:
    """Classe de base pour les écoulements potentiels 2D."""

    def __init__(self, nom: str = "PlanPlus"):
        # Initialisation des dérivées partielles
        self.d_phi_y = None
        self.d_phi_x = None
        # Initialisation de la fonction complèxe
        self.f = 0
        # Déclaration des variables symboliques
        self.x = sympy.symbols("x", real=True)
        self.y = sympy.symbols("y", real=True)
        self.z = self.x + sympy.I * self.y
        self.t = sympy.symbols("t", real=True)
        self.nom = nom

    def __add__(self, other):
        """Addition d'écoulements (potentiels complexes)."""
        f = self.f + other.f
        new_field = PlanPlus()
        new_field.f = f
        new_field.nom = f"{self.nom}, {other.nom}"
        new_field.update()
        return new_field

    def update(self):
        """Met à jour les dérivées partielles du potentiel."""
        phi_expr = self.phi()
        self.d_phi_x = sympy.diff(phi_expr, self.x)
        self.d_phi_y = sympy.diff(phi_expr, self.y)

    def phi(self):
        """Fonction potentielle."""
        return sympy.re(self.f)

    def psi(self):
        """Fonction de courant."""
        return sympy.im(self.f)

    def c_niv1(self, const):
        """Courbe de niveau de la fonction de courant."""
        return sympy.Eq(self.psi(), const)

    def c_niv2(self, const):
        """Courbe de niveau de la fonction potentielle."""
        return sympy.Eq(self.phi(), const)

    def vect_v(self):
        """Champ de vitesse."""
        return sympy.Matrix([self.d_phi_x, self.d_phi_y])

    def __str__(self):
        return f"Ecoulement : {self.nom}"

    def __repr__(self):
        return self.__str__()


class EUniforme(PlanPlus):
    """Ecoulement uniforme."""

    def __init__(self, u: float = 1):
        """
        :param u: la vitesse (m/s)
        """
        super().__init__()
        self.U = u
        self.f = self.U * self.z
        self.nom = "uniforme"
        self.update()


class Puits(PlanPlus):
    """Cette classe implémente un puits
    """

    def __init__(self, a: float = 1, qv: float = 1):
        super().__init__()
        self.a = a
        self.Qv = qv
        self.f = -qv * sympy.log(self.z - a) / (2 * sympy.pi)
        self.nom = "puits"
        self.update()


class Source(PlanPlus):
    """Cette classe implémente une
    source
    """

    def __init__(self, a: float = 1, qv: float = 1):
        super().__init__()
        self.a = a
        self.Qv = qv
        self.f = qv * sympy.log(self.z + a) / (2 * sympy.pi)
        self.nom = "source"
        self.update()


class Tourbillon(PlanPlus):
    """Cette classe est une proposition
    de l'intelligence artificielle
    ChatGPT afin de tenir compte d'un
    comporte un peu plus réaliste"""
    def __init__(self, gamma=1.5):
        super().__init__()
        self.gamma = gamma
        self.f = gamma / (2 * sympy.pi * sympy.I) * sympy.log(self.z)
        self.nom = "tourbillon"
        self.update()


if __name__ == "__main__":
    E1 = EUniforme(15)
    E2 = Puits(1, 3)
    E3 = Source(1, 3)
    E4 = Tourbillon(gamma=1.5)

    E = E1 + E2 + E3 + E4

    try:
        with open("ProjetMecaniqueFluide.tex", "w") as tex:
            tex.write("\\documentclass{article}\n"
                      "\\usepackage{amsmath}\n"
                      "\\begin{document}\n")
            tex.write(latex(E1))
            tex.write(latex(E2))
            tex.write(latex(E))
            tex.write("\\end{document}\n")
    except IOError as e:
        print(f"Erreur lors de l'écriture du fichier LaTeX : {e}")

    fig1 = courant(E, 3, 3, np.linspace(-10, 10, 5))
    # fig2 = potentiel(E, 3, 3, np.linspace(-10, 10, 50))
    # champ(E, (-2, 2), (-1, 1))

    plt.show()
