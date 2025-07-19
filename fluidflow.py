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
    """Trace les courbes de courant pour plusieurs niveaux."""
    figures = []
    for n in niveaux:
        fig = sympy.plot_implicit(objet.c_niv1(n), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False,
                                  adaptive=False, depth=4)
        figures.append(fig)
    # Fusionner les figures
    fig_final = figures[0]
    for f in figures[1:]:
        fig_final.extend(f)
    fig_final.title = "Fonction de courant de l'écoulement"
    fig_final.xlabel = "X"
    fig_final.ylabel = "Y"
    return fig_final


def potentiel(objet, a, b, niveaux):
    """Trace les courbes de potentiel pour plusieurs niveaux."""
    figures = []
    for n in niveaux:
        fig = sympy.plot_implicit(objet.c_niv2(n), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False)
        figures.append(fig)
    fig_final = figures[0]
    for f in figures[1:]:
        fig_final.extend(f)
    fig_final.title = "Fonction potentielle de l'écoulement"
    fig_final.xlabel = "X"
    fig_final.ylabel = "Y"
    return fig_final


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
        new_field.nom = f"{self.nom} et {other.nom} combinés"
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


if __name__ == "__main__":
    E1 = EUniforme(7)
    E2 = Puits(1, 2)
    E3 = Source(2, 2.5)

    E4 = E3 + E2
    E4.nom = "Doublet"
    E = E1 + E4

    fig1 = courant(E, 5, 5, (1, -1, -2, 2, 3, -3))
    # fig2 = potentiel(E, 5, 5, (1, -1, -2, 2, 3, -3))
    # champ(E4, (-5, 5), (-2, 2))

    fig1.show()
    # fig2.show()
    # Export LaTeX
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
