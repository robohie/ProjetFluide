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
    text = (
        f"\\section*{{Ecoulement : {objet.nom}}}\n"
        f"Potentiel complexe \\[{sympy.latex(objet.f)}\\]\n"
        f"Fonction potentielle \\[{sympy.latex(objet.phi())}\\]\n"
        f"Fonction courant \\[{sympy.latex(objet.psi())}\\]\n"
        f"Champ vectoriel \\[{sympy.latex(objet.vect_v())}\\]\n"
    )
    return text


def courant(objet, a, b):
    """Cette fonction sert à représenter
    la fonction de courant
    """
    figure1 = sympy.plot_implicit(objet.c_niv1(-3), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False, adaptive=False, depth=4)
    figure2 = sympy.plot_implicit(objet.c_niv1(-2), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False, adaptive=False, depth=4)
    figure3 = sympy.plot_implicit(objet.c_niv1(-1), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False, adaptive=False, depth=4)
    figure4 = sympy.plot_implicit(objet.c_niv1(1), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False, adaptive=False, depth=4)
    figure5 = sympy.plot_implicit(objet.c_niv1(2), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False, adaptive=False, depth=4)
    figure6 = sympy.plot_implicit(objet.c_niv1(3), (objet.x, -a, a),
                                  (objet.y, -b, b), show=False, adaptive=False, depth=4)
    figure2.extend(figure1)
    figure3.extend(figure2)
    figure4.extend(figure3)
    figure5.extend(figure4)
    figure6.extend(figure5)

    figure6.title = "Fonction de courant de l'écoulement"

    return figure6


def potentiel(objet, a, b):
    """Cette fonction sert à représenter
    la fonction potentielle
    :param objet:
    :param a:
    :param b:
    """
    figure1 = sympy.plot_implicit(objet.c_niv2(-3),
                                  (objet.x, -a, a),
                                  (objet.y, -b, b), show=False)
    figure2 = sympy.plot_implicit(objet.c_niv2(-2),
                                  (objet.x, -a, a),
                                  (objet.y, -b, b), show=False)
    figure3 = sympy.plot_implicit(objet.c_niv2(-1),
                                  (objet.x, -a, a),
                                  (objet.y, -b, b), show=False)
    figure4 = sympy.plot_implicit(objet.c_niv2(0),
                                  (objet.x, -a, a),
                                  (objet.y, -b, b), show=False)
    figure5 = sympy.plot_implicit(objet.c_niv2(1),
                                  (objet.x, -a, a),
                                  (objet.y, -b, b), show=False)
    figure6 = sympy.plot_implicit(objet.c_niv2(2),
                                  (objet.x, -a, a),
                                  (objet.y, -b, b), show=False)

    figure2.extend(figure1)
    figure3.extend(figure2)
    figure4.extend(figure3)
    figure5.extend(figure4)
    figure6.extend(figure5)

    figure6.title = "Fonction potentielle de l'écoulement"
    figure6.xlabel = "X"
    figure6.ylabel = "Y"

    return figure6


def champ(objet, x_lim=(-10, 10), y_lim=(-10, 10), density=20):
    """Cette fonction sert à représenter
    le champ vectoriel de vitesse
    :param objet:
    :param x_lim:
    :param y_lim:
    :param density:
    """
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
    plt.show()


class PlanPlus:
    """Cette classe représente
    un système d'axes rectangulaires
    Elle permet de définir les
    coordonnées d'un point dans
    le plan

    La classe contient aussi des
    choses liées aux écoulements
    en général
    """

    def __init__(self):
        self.d_phi_y = None
        self.d_phi_x = None
        self.f = 0
        self.x = sympy.symbols("x", real=True)
        self.y = sympy.symbols("y", real=True)
        self.z = self.x + sympy.I * self.y
        self.t = sympy.symbols("t", real=True)
        self.nom = "PlanPlus"

    def __add__(self, other):
        f = self.f + other.f
        new_field = PlanPlus()
        new_field.f = f
        new_field.nom = (self.nom + ' et '
                         + other.nom + ' combinés')
        new_field.update()
        return new_field

    def update(self):
        phi_expr = self.phi()
        self.d_phi_x = sympy.diff(phi_expr, self.x)
        self.d_phi_y = sympy.diff(phi_expr, self.y)

    def phi(self):
        """Cette méthode retourne la
        fonction courant de l'écoulement
        """
        return sympy.sympify(sympy.re(self.f))

    def psi(self):
        """Cette méthode retourne la
        fonction potentielle
        """
        return sympy.sympify(sympy.im(self.f))

    def c_niv1(self, const):
        """Cette méthode retourne une
         courbe de 2D de l'écoulement

        :param const: constante
        :return: sympy.core.relational.Equality
        """
        niveau = sympy.Eq(self.psi(), const)
        return niveau

    def c_niv2(self, const):
        """Cette méthode retourne une
        courbe de niveau de la fonction
        potentielle

        :param const: constante
        :return: sympy.core.relational.Equality
        """
        niveau = sympy.Eq(self.phi(), const)
        return niveau

    def vect_v(self):
        """Cette méthode calcul le champ
        de vitesse de l'écoulement
        """
        return sympy.sympify(sympy.Matrix([self.d_phi_x, self.d_phi_y]))


class EUniforme(PlanPlus):
    """Cette classe implémente
    un écoulement fluide uniforme"""

    def __init__(self, u: float = 1):
        """

        :param u: la vitesse (m/s)
        """
        super().__init__()
        self.U = u
        self.f = self.U * self.z
        self.nom = "uniforme"
        self.update()

    def __str__(self):
        pass

    def __repr__(self):
        pass


class Doublet(PlanPlus):
    """Cette classe implémente un
    doublet i.e. combili source
    et puits
    """

    def __init__(self, a: float = 1, qv: float = 1):
        """

        :param a: distance entre la source
                  (ou le puits) et l'origine
        :param qv: débit volumique
        """
        super().__init__()
        self.f = qv * sympy.log((self.z + a) / (self.z - a)) / (2 * sympy.pi)
        self.a = a
        self.Qv = qv
        self.f = qv * sympy.log((self.z + a) / (self.z - a)) / (2 * sympy.pi)
        self.nom = "doublet"
        self.update()

    def __str__(self):
        pass

    def __repr__(self):
        pass


if __name__ == "__main__":
    E1 = EUniforme(5)
    E2 = Doublet(0.05, 0.5)

    E = E1 + E2

    fig1 = courant(E, 5, 5)

    champ(E, (-0.5, 0.5), (-0.5, 0.5))
    with open(f"C:\\Users\\itel\\Documents\\ProjetMecaniqueFluide.tex",
              "w") as tex:
        tex.write(f"\\documentclass{{article}}\n"
                  f"\\usepackage{{amsmath}}\n"
                  f"\\begin{{document}}\n")
        tex.write(latex(E))
        tex.write(f"\\end{{document}}")
