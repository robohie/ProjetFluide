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
#     ESAFE ISIMO BENJAMIN (GC)
#     KABONGO MUKENDI ODON (GM)
#     MUKENGE KOLM THADDEE (GEI)
#     TSHIMANGA KABANZA CRIS-BOAZ (GEI)
#     DEBORAH (GC)
# ================================

import sympy
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional


def latex(objet) -> str:
    """Génère le code LaTeX décrivant un écoulement."""
    text = (
        f"\\section*{{Écoulement : {objet.nom}}}\n"
        f"Potentiel complexe : \\[{sympy.latex(objet.f)}\\]\n"
        f"Fonction potentielle : \\[{sympy.latex(objet.phi())}\\]\n"
        f"Fonction courant : \\[{sympy.latex(objet.psi())}\\]\n"
        f"Champ vectoriel : \\[{sympy.latex(objet.vect_v())}\\]\n"
    )
    return text


def plot_niveaux(objet, func, niveaux, a, b, titre: str) -> sympy.plotting.plot:
    """Trace plusieurs courbes de niveau pour une fonction donnée."""
    figures = []
    for n in niveaux:
        fig = sympy.plot_implicit(func(n), (objet.x, -a, a), (objet.y, -b, b),
                                  show=False, adaptive=False, depth=4)
        figures.append(fig)
    # Fusionne toutes les figures
    for i in range(1, len(figures)):
        figures[i].extend(figures[i - 1])
    figures[-1].title = titre
    return figures[-1]


def courant(objet, a, b):
    niveaux = [-3, -2, -1, 1, 2, 3]
    return plot_niveaux(objet, objet.c_niv1, niveaux, a, b,
                        "Fonction de courant de l'écoulement")


def potentiel(objet, a, b):
    niveaux = [-3, -2, -1, 0, 1, 2]
    return plot_niveaux(objet, objet.c_niv2, niveaux, a, b,
                        "Fonction potentielle de l'écoulement")


def champ(objet, x_lim: Tuple[float, float] = (-10, 10), y_lim: Tuple[float, float] = (-10, 10),
          density: int = 20, save_path: Optional[str] = None):
    """Trace le champ vectoriel de vitesse."""
    X, Y = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], density),
        np.linspace(y_lim[0], y_lim[1], density)
    )
    U = sympy.lambdify((objet.x, objet.y), objet.d_phi_x)(X, Y)
    V = sympy.lambdify((objet.x, objet.y), objet.d_phi_y)(X, Y)
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V)
    plt.grid(True)
    plt.title("Champ de vitesse")
    plt.xlabel("X")
    plt.ylabel("Y")
    if save_path:
        plt.savefig(save_path)
    plt.show()


class PlanPlus:
    """Système de coordonnées plan + outils d'écoulement."""

    def __init__(self):
        self.d_phi_y = None
        self.d_phi_x = None
        self.f = 0
        self.x, self.y = sympy.symbols("x y", real=True)
        self.z = self.x + sympy.I * self.y
        self.t = sympy.symbols("t", real=True)
        self.nom = "PlanPlus"
        self.update()

    def __add__(self, other):
        f = self.f + other.f
        new_field = PlanPlus()
        new_field.f = f
        new_field.nom = f"{self.nom} et {other.nom} combinés"
        new_field.update()
        return new_field

    def update(self):
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
        """Courbe de niveau pour la fonction courant."""
        return sympy.Eq(self.psi(), const)

    def c_niv2(self, const):
        """Courbe de niveau pour la fonction potentielle."""
        return sympy.Eq(self.phi(), const)

    def vect_v(self):
        """Champ de vitesse."""
        return sympy.Matrix([self.d_phi_x, self.d_phi_y])

    def __str__(self):
        return f"Écoulement: {self.nom}"

    def __repr__(self):
        return f"<PlanPlus nom={self.nom}>"


class EUniforme(PlanPlus):
    """Écoulement fluide uniforme."""

    def __init__(self, u: float = 1):
        super().__init__()
        self.U = u
        self.f = self.U * self.z
        self.nom = "uniforme"
        self.update()

    def __str__(self):
        return f"Écoulement uniforme (U={self.U})"

    def __repr__(self):
        return f"<EUniforme U={self.U}>"


class Doublet(PlanPlus):
    """Doublet (source + puits)."""

    def __init__(self, a: float = 1, qv: float = 1):
        super().__init__()
        self.a = a
        self.Qv = qv
        self.f = qv * sympy.log((self.z + a) / (self.z - a)) / (2 * sympy.pi)
        self.nom = "doublet"
        self.update()

    def __str__(self):
        return f"Doublet (a={self.a}, Qv={self.Qv})"

    def __repr__(self):
        return f"<Doublet a={self.a}, Qv={self.Qv}>"


if __name__ == "__main__":
    # Paramètres
    E1 = EUniforme(5)
    E2 = Doublet(0.05, 0.5)
    E = E1 + E2

    # Tracé des courbes de courant
    fig1 = courant(E, 5, 5)
    fig1.show()

    # Tracé du champ de vitesse
    champ(E, (-0.5, 0.5), (-0.5, 0.5))

    # Génération du fichier LaTeX
    latex_path = "C:\\Users\\itel\\Documents\\ProjetMecaniqueFluide.tex"
    with open(latex_path, "w") as tex:
        tex.write("\\documentclass{article}\n"
                  "\\usepackage{amsmath}\n"
                  "\\begin{document}\n")
        tex.write(latex(E))
        tex.write("\\end{document}")
    print(f"LaTeX exporté sous {latex_path}")
