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
from pathlib import Path
from typing import List, Tuple, Optional
import warnings

# Supprimer les avertissements de division par zéro
warnings.filterwarnings("ignore", category=RuntimeWarning)


def latex(objet) -> str:
    """Génère une section LaTeX détaillée résumant l'écoulement."""
    return (
        f"\\section*{{Ecoulement : {objet.nom}}}\n"
        f"\\textbf{{Potentiel complexe}}: \\[{sympy.latex(objet.f.simplify())}\\]\n"
        f"\\textbf{{Fonction potentielle}}: \\[{sympy.latex(objet.phi().simplify())}\\]\n"
        f"\\textbf{{Fonction courant}}: \\[{sympy.latex(objet.psi().simplify())}\\]\n"
        f"\\textbf{{Champ vectoriel}}: \\[{sympy.latex(objet.vect_v().simplify())}\\]\n"
        f"\\textbf{{Paramètres}}: {objet.latex_params()}\n"
    )


def courant(objet, a: float, b: float, niveaux: List[float],
            title: str = "Lignes de courant") -> plt.Figure:
    """Trace les lignes de courant avec gestion des singularités."""
    x_vals = np.linspace(-a, a, 400)
    y_vals = np.linspace(-b, b, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calcul de la fonction courant
    if isinstance(objet, ProfilJoukowski):
        Psi = objet.eval_psi(X, Y)
    else:
        psi_func = sympy.lambdify((objet.x, objet.y), objet.psi(), "numpy", cse=True)
        Psi = psi_func(X, Y)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé des lignes de courant
    contour = ax.contour(X, Y, Psi, levels=sorted(niveaux), linewidths=1.5)
    plt.clabel(contour, inline=True, fontsize=8)

    # Tracé du profil pour Joukowski
    if isinstance(objet, ProfilJoukowski):
        ax.plot(objet.profil_x, objet.profil_y, 'r-', linewidth=2.5, label='Profil')

    # Configuration du graphique
    ax.set_title(f"{title} ({objet.nom})", fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axis('equal')
    ax.legend(loc='best')

    return fig


def potentiel(objet, a: float, b: float, niveaux: List[float],
              title: str = "Équipotentiels") -> plt.Figure:
    """Trace les équipotentielles avec optimisation des calculs."""
    x_vals = np.linspace(-a, a, 400)
    y_vals = np.linspace(-b, b, 400)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Calcul de la fonction potentielle
    if isinstance(objet, ProfilJoukowski):
        Phi = objet.eval_phi(X, Y)
    else:
        phi_func = sympy.lambdify((objet.x, objet.y), objet.phi(), "numpy", cse=True)
        Phi = phi_func(X, Y)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé des équipotentielles
    contour = ax.contour(X, Y, Phi, levels=sorted(niveaux), linewidths=1.5)
    plt.clabel(contour, inline=True, fontsize=8)

    # Configuration du graphique
    ax.set_title(f"{title} ({objet.nom})", fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axis('equal')

    return fig


def champ(objet, x_lim: Tuple[float, float] = (-5, 5),
          y_lim: Tuple[float, float] = (-5, 5),
          density: int = 15) -> plt.Figure:
    """Affiche le champ vectoriel de vitesse avec normalisation."""
    # Création de la grille
    X, Y = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], density),
        np.linspace(y_lim[0], y_lim[1], density)
    )

    # Calcul des composantes de vitesse
    dphi_x_func = sympy.lambdify((objet.x, objet.y), objet.d_phi_x, "numpy")
    dphi_y_func = sympy.lambdify((objet.x, objet.y), objet.d_phi_y, "numpy")

    U = dphi_x_func(X, Y)
    V = dphi_y_func(X, Y)

    # Calcul de la magnitude pour la normalisation
    magnitude = np.sqrt(U ** 2 + V ** 2)
    U_norm = np.where(magnitude > 0, U / magnitude, 0)
    V_norm = np.where(magnitude > 0, V / magnitude, 0)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracé du champ vectoriel
    ax.quiver(X, Y, U_norm, V_norm, magnitude,
              cmap='viridis', scale=25, width=0.005)

    # Tracé du profil pour Joukowski
    if isinstance(objet, ProfilJoukowski):
        ax.plot(objet.profil_x, objet.profil_y, 'r-', linewidth=2.5)

    # Configuration du graphique
    ax.set_title(f"Champ de vitesse ({objet.nom})", fontsize=14)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')

    return fig


class PlanPlus:
    """Classe de base pour les écoulements potentiels 2D."""

    def __init__(self, nom: str = "PlanPlus"):
        self.x = sympy.symbols("x", real=True)
        self.y = sympy.symbols("y", real=True)
        self.z = self.x + sympy.I * self.y
        self.t = sympy.symbols("t", real=True)
        self.nom = nom
        self.f = 0  # Potentiel complexe
        self._update_derivatives()

    def _update_derivatives(self):
        """Met à jour les dérivées partielles du potentiel."""
        phi_expr = self.phi()
        self.d_phi_x = sympy.diff(phi_expr, self.x)
        self.d_phi_y = sympy.diff(phi_expr, self.y)

    def __add__(self, other):
        """Addition d'écoulements (superposition des potentiels complexes)."""
        new_field = PlanPlus(f"{self.nom} + {other.nom}")
        new_field.f = self.f + other.f
        new_field._update_derivatives()
        return new_field

    def phi(self) -> sympy.Expr:
        """Fonction potentielle (partie réelle)."""
        return sympy.re(self.f)

    def psi(self) -> sympy.Expr:
        """Fonction de courant (partie imaginaire)."""
        return sympy.im(self.f)

    def vect_v(self) -> sympy.Matrix:
        """Champ de vitesse vectoriel."""
        return sympy.Matrix([self.d_phi_x, self.d_phi_y])

    def latex_params(self) -> str:
        """Retourne les paramètres au format LaTeX."""
        return "Aucun paramètre spécifique"

    def __str__(self):
        return f"Ecoulement: {self.nom}"

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.nom}>"


class EUniforme(PlanPlus):
    """Écoulement uniforme dans une direction donnée."""

    def __init__(self, u: float = 1.0, angle: float = 0.0):
        """
        :param u: Magnitude de la vitesse (m/s)
        :param angle: Angle de direction en radians
        """
        super().__init__(f"Uniforme (U={u}, α={np.degrees(angle):.1f}°)")
        self.U = u
        self.angle = angle
        self.f = self.U * sympy.exp(-sympy.I * angle) * self.z
        self._update_derivatives()

    def latex_params(self) -> str:
        return f"$U_\\infty = {self.U}$ m/s, $\\alpha = {np.degrees(self.angle):.2f}^\\circ$"


class Source(PlanPlus):
    """Source avec débit volumétrique donné."""

    def __init__(self, debit: float = 1.0, position: complex = 0.0):
        """
        :param debit: Débit volumétrique (m³/s)
        :param position: Position de la source (complex)
        """
        super().__init__(f"Source (Q={debit})")
        self.Q = debit
        self.position = position
        self.f = (self.Q / (2 * sympy.pi)) * sympy.log(self.z - position)
        self._update_derivatives()

    def latex_params(self) -> str:
        return f"$Q = {self.Q}$ m³/s, Position $= {self.position}$"


class Puits(Source):
    """Puits avec débit volumétrique donné."""

    def __init__(self, debit: float = 1.0, position: complex = 0.0):
        super().__init__(-debit, position)
        self.nom = f"Puits (Q={debit})"


class Tourbillon(PlanPlus):
    """Tourbillon avec circulation donnée."""

    def __init__(self, circulation: float = 1.0, position: complex = 0.0):
        """
        :param circulation: Circulation (m²/s)
        :param position: Position du tourbillon (complex)
        """
        super().__init__(f"Tourbillon (Γ={circulation})")
        self.Gamma = circulation
        self.position = position
        self.f = (sympy.I * self.Gamma / (2 * sympy.pi)) * sympy.log(self.z - position)
        self._update_derivatives()

    def latex_params(self) -> str:
        return f"$\\Gamma = {self.Gamma}$ m²/s, Position $= {self.position}$"


class Dipole(PlanPlus):
    """Dipole avec moment dipolaire donné."""

    def __init__(self, moment: complex = 1.0, position: complex = 0.0):
        """
        :param moment: Moment dipolaire complexe
        :param position: Position du dipole
        """
        super().__init__(f"Dipole (μ={moment})")
        self.moment = moment
        self.position = position
        self.f = moment / (2 * sympy.pi * (self.z - position))
        self._update_derivatives()

    def latex_params(self) -> str:
        return f"$\\mu = {self.moment}$, Position $= {self.position}$"


class ProfilJoukowski(PlanPlus):
    """Écoulement autour d'un profil d'aile par transformation de Joukowski."""

    def __init__(self, u_inf: float = 1.0, alpha: float = 0.0,
                 c: float = 1.0, x0: float = -0.1, y0: float = 0.1,
                 gamma: Optional[float] = None):
        """
        :param u_inf: Vitesse à l'infini
        :param alpha: Angle d'attaque (radians)
        :param c: Paramètre de la transformation
        :param x0: Décalage x du centre du cercle
        :param y0: Décalage y du centre du cercle
        :param gamma: Circulation (si None, calculée par condition Kutta)
        """
        super().__init__("Profil Joukowski")
        self.c = c
        self.x0 = x0
        self.y0 = y0
        self.U_inf = u_inf
        self.alpha = alpha

        # Calcul du rayon du cercle
        self.R = np.sqrt((x0 + c) ** 2 + y0 ** 2)

        # Calcul de la position du centre
        self.z0 = x0 + y0 * 1j

        # Condition de Kutta (si gamma non spécifié)
        beta = np.arcsin(y0 / self.R)
        self.gamma = gamma or -4 * np.pi * u_inf * self.R * np.sin(alpha + beta)

        # Génération du profil
        self._generer_profil()

        # Mise à jour du potentiel complexe
        self._update_potentiel()
        self._update_derivatives()

    def _generer_profil(self):
        """Génère les coordonnées du profil d'aile."""
        theta = np.linspace(0, 2 * np.pi, 300)
        z_c = self.z0 + self.R * np.exp(1j * theta)
        zeta = z_c + self.c ** 2 / z_c
        self.profil_x = np.real(zeta)
        self.profil_y = np.imag(zeta)

    def _update_potentiel(self):
        """Met à jour le potentiel complexe."""
        # Écoulement uniforme incliné
        e_uniforme = self.U_inf * (
                sympy.exp(-sympy.I * self.alpha) * (self.z - self.z0) +
                (self.R ** 2 * sympy.exp(sympy.I * self.alpha)) / (self.z - self.z0)
        )

        # Tourbillon au centre
        tourbillon = (sympy.I * self.gamma / (2 * sympy.pi)) * sympy.log(self.z - self.z0)

        # Transformation de Joukowski
        self.f = (e_uniforme + tourbillon).subs(
            self.z, self.z + self.c ** 2 / self.z
        )

    def _transformation_inverse(self, zeta: np.ndarray) -> np.ndarray:
        """Transforme les points du plan physique vers le plan du cercle."""
        return 0.5 * (zeta + np.sqrt(zeta ** 2 - 4 * self.c ** 2))

    def eval_phi(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Évalue la fonction potentielle sur une grille."""
        Z = x + 1j * y
        Zc = self._transformation_inverse(Z)

        # Calcul du potentiel dans le plan du cercle
        W = self.U_inf * (
                np.exp(-1j * self.alpha) * (Zc - self.z0) +
                (self.R ** 2 * np.exp(1j * self.alpha)) / (Zc - self.z0)
        ) + (1j * self.gamma / (2 * np.pi)) * np.log(Zc - self.z0)

        return np.real(W)

    def eval_psi(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Évalue la fonction courant sur une grille."""
        return np.imag(self.eval_phi(x, y) + 0j)  # +0j pour forcer le type complexe

    def latex_params(self) -> str:
        params = [
            f"$U_\\infty = {self.U_inf}$",
            f"$\\alpha = {np.degrees(self.alpha):.2f}^\\circ$",
            f"$c = {self.c}$",
            f"$x_0 = {self.x0}$",
            f"$y_0 = {self.y0}$",
            f"$R = {self.R:.3f}$",
            f"$\\Gamma = {self.gamma:.4f}$"
        ]
        return ", ".join(params)


def export_plots(objet, a: float, b: float, prefix: str = "ecoulement"):
    """Exporte toutes les visualisations dans des fichiers."""
    # Création du dossier de sortie
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Génération des figures
    fig_courant = courant(objet, a, b, list(np.linspace(-2, 2, 15)))
    fig_potentiel = potentiel(objet, a, b, list(np.linspace(-2, 2, 15)))
    fig_champ = champ(objet, (-a, a), (-b, b))

    # Sauvegarde des figures
    fig_courant.savefig(output_dir / f"{prefix}_courant.png", dpi=300)
    fig_potentiel.savefig(output_dir / f"{prefix}_potentiel.png", dpi=300)
    fig_champ.savefig(output_dir / f"{prefix}_champ.png", dpi=300)

    # Fermeture des figures pour libérer la mémoire
    plt.close(fig_courant)
    plt.close(fig_potentiel)
    plt.close(fig_champ)

    return str(output_dir)


def generate_report(objet, filename: str = "ProjetMecaniqueFluide.tex"):
    """Génère un rapport LaTeX complet avec les résultats."""
    try:
        with open(filename, "w") as tex_file:
            tex_file.write(
                "\\documentclass{article}\n"
                "\\usepackage{amsmath}\n"
                "\\usepackage{graphicx}\n"
                "\\usepackage{geometry}\n"
                "\\geometry{a4paper, margin=1.5cm}\n"
                "\\title{Projet de Mécanique des Fluides}\n"
                "\\date{\\today}\n"
                "\\begin{document}\n"
                "\\maketitle\n"
            )

            # Description de l'écoulement
            tex_file.write(latex(objet))

            # Ajout des figures
            if isinstance(objet, ProfilJoukowski):
                export_dir = export_plots(objet, 3, 3, "profil")
                tex_file.write(
                    "\\begin{figure}[ht]\n"
                    "\\centering\n"
                    "\\includegraphics[width=0.8\\textwidth]{" +
                    export_dir + "/profil_courant.png}\n"
                                 "\\caption{Lignes de courant autour du profil}\n"
                                 "\\end{figure}\n\n"
                                 "\\begin{figure}[ht]\n"
                                 "\\centering\n"
                                 "\\includegraphics[width=0.8\\textwidth]{" + export_dir + "/profil_potentiel.png}\n"
                                                                                           "\\caption{Équipotentielles autour du profil}\n"
                                                                                           "\\end{figure}\n\n"
                                                                                           "\\begin{figure}[ht]\n"
                                                                                           "\\centering\n"
                                                                                           "\\includegraphics[width=0.8\\textwidth]{" + export_dir + "/profil_champ.png}\n"
                                                                                                                                                     "\\caption{Champ de vitesse autour du profil}\n"
                                                                                                                                                     "\\end{figure}\n"
                )

            tex_file.write("\\end{document}\n")

        print(f"Rapport généré avec succès: {filename}")
        return True

    except Exception as e:
        print(f"Erreur lors de la génération du rapport: {str(e)}")
        return False


if __name__ == "__main__":
    # Exemple d'écoulement uniforme
    ecoulement_uniforme = EUniforme(u=1.5, angle=np.radians(10))

    # Exemple de source et puits
    source = Source(debit=2.0, position=-2 + 1j)
    puits = Puits(debit=2.0, position=2 + 1j)

    # Combinaison d'écoulements
    ecoulement_combine = ecoulement_uniforme + source + puits

    # Profil de Joukowski
    profil = ProfilJoukowski(
        u_inf=1.0,
        alpha=np.radians(5),
        c=1.0,
        x0=-0.1,
        y0=0.1
    )

    # Génération du rapport pour le profil
    generate_report(profil)

    # Visualisation interactive
    courant(profil, 3, 3, list(np.linspace(-2, 2, 15))).suptitle("Profil de Joukowski")
    plt.tight_layout()
    plt.show()
