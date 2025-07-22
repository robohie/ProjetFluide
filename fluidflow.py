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

    if isinstance(objet, ProfilJoukowski):
        # Evaluation spéciale pour Joukowski
        Psi = objet.eval_psi(X, Y)
    else:
        psi = sympy.lambdify((objet.x, objet.y), objet.psi(), "numpy")
        Psi = psi(X, Y)

    figure, axe = plt.subplots(figsize=(8, 6))

    axe.contour(X, Y, Psi, levels=niveaux)

    # Tracer les lignes de courant
    if isinstance(objet, ProfilJoukowski):
        axe.plot(objet.profil_x, objet.profil_y, "r-", linewidth=2)

    axe.set_title(label=f"lignes de courant ({objet.nom})")
    axe.grid(True)
    axe.set_aspect("equal")

    return figure


def potentiel(objet, a, b, niveaux):
    """Fonction qui trace les équipotentiels
    """
    niveaux = sorted(niveaux)
    x_vals = np.linspace(-a, a, 400)
    y_vals = np.linspace(-b, b, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    if isinstance(objet, ProfilJoukowski):
        Phi = objet.eval_phi(X, Y)
    else:
        phi = sympy.lambdify((objet.x, objet.y), objet.phi(), "numpy")
        Phi = phi(X, Y)

    figure, axe = plt.subplots(figsize=(8, 6))

    axe.contour(X, Y, Phi, levels=niveaux)
    axe.set_title(label=f"Equipotentiels ({objet.nom})")
    axe.grid(True)
    axe.set_aspect("equal")

    return figure


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


class ProfilJoukowski(PlanPlus):
    """Classe pour simuler l'écoulement
    l'écoulement autour d'un profil NACA
    à l'aide de la transformation de
    Joukowski
    """

    def __init__(self, u_inf=1.0, alpha=0.0, c=1.0, x0=-0.1, y0=0.1, gamma=None):
        """Ce code est une proposition de DeepSeek
        pour tenir compte du cas d'un profil réel
        comme le NACA

        :param u_inf: vitesse à l'infini
        :param alpha: angle d'attaque en radians
        :param c: Paramètre de la transformation
        :param x0: Décalage en x du centre du cercle
        :param y0: Décalage en y du centre du cercle
        :param gamma: lié à la condition de Kuta
        """
        super().__init__("Profil")

        # Paramètres de la transformation
        self.c = c
        self.x0 = x0
        self.y0 = y0

        # Calcul du rayon du cercle pour passer par (c, 0)
        self.R = np.sqrt((x0 + c) ** 2 + y0 ** 2)

        # Paramètres de l'écoulement
        self.U_inf = u_inf
        self.alpha = alpha

        # Circulation
        if gamma is None:
            self.gamma = (-4 * np.pi * u_inf * self.R *
                          np.sin(alpha + np.arcsin(y0 / self.R)))
        else:
            self.gamma = gamma

        # potentiel complexe dans le plan du cercle
        z0 = x0 + y0 * sympy.I
        self.z0 = x0 + y0 * 1j

        # Ecoulement uniforme incliné autour du cercle
        terme1 = u_inf * (sympy.exp(-sympy.I * alpha) * (self.z - z0) +
                          (self.R ** 2 * sympy.exp(sympy.I * alpha)) /
                          self.z - z0)

        # Tourbillon au centre du cercle
        terme2 = (sympy.I * self.gamma) / (2 * sympy.pi) * sympy.log(self.z - z0)

        self.f_circle = terme1 + terme2

        # Transformation de Joukowski
        self.f = self.f_circle.subs(self.z, self.z + self.c ** 2 / self.z)

        # Générer les points du profil
        self._generer_profil()

        self.update()

    def _generer_profil(self):
        """Génère les coordonnées du profil Joukowski"""
        theta = np.linspace(0, 2 * np.pi, 200)
        z_c = self.x0 + 1j * self.y0 + self.R * np.exp(1j * theta)
        zeta = z_c + self.c ** 2 / z_c
        self.profil_x = np.real(zeta)
        self.profil_y = np.imag(zeta)

    def f_cercle(self, z):
        """Potentiel complexe dans le plan du cercle
        (évaluation numérique)
        """
        terme2 = (1j * self.gamma) / (2 * np.pi) * np.log(z - self.z0)
        terme1 = self.U_inf * (np.exp(-1j * self.alpha) * (z - self.z0) +
                               (self.R ** 2 * np.exp(1j * self.alpha)) /
                               z - self.z0)
        return terme1 + terme2

    def eval_phi(self, x, y):
        """Evaluation numérique de la fonction potentielle"""
        zeta = x + 1j * y
        # Transformation inverse de Joukowski
        with np.errstate(all="ignore"):
            z_c = (zeta + np.sqrt(zeta ** 2 - 4 * self.c ** 2)) / 2

        # Evaluation du potentiel complexe
        f_val = self.f_cercle(z_c)
        return np.real(f_val)

    def eval_psi(self, x, y):
        """Evaluation numérique de la fonction de courant"""
        zeta = x + 1j * y
        # Transformation inverse de Joukowski
        with np.errstate(all="ignore"):
            z_c = (zeta + np.sqrt(zeta ** 2 - 4 * self.c ** 2)) / 2

        # Evaluation du potentiel complexe
        f_val = self.f_cercle(z_c)

        return np.imag(f_val)


if __name__ == "__main__":
    P1 = ProfilJoukowski(
        u_inf=1.0,
        alpha=np.radians(5),
        c=1.0,
        x0=-0.1,
        y0=0.1
    )

    # Tracé des lignes de courant
    fig = courant(P1, 3, 3, np.linspace(-2, 2, 10))
    plt.show()

    try:
        with open("ProjetMecaniqueFluide.tex", "w") as tex:
            tex.write("\\documentclass{article}\n"
                      "\\usepackage{amsmath}\n"
                      "\\begin{document}\n")
            tex.write(latex(P1))
            tex.write("\\end{document}\n")
    except IOError as e:
        print(f"Erreur lors de l'écriture du fichier LaTeX : {e}")
