import tkinter as tk
from tkinter import ttk

# Fonction de l'action du bouton de menu
def menu_toggle():
    print("Menu button clicked")

# Création de la fenêtre principale
window = tk.Tk()
window.title("MoodIT")

# Création du conteneur du menu
header_container = ttk.Frame(window)
header_container.pack()

# Création du bouton de menu avec un fond transparent
menu_toggle = ttk.Button(header_container, text="Menu", command=menu_toggle, style="Transparent.TButton")
menu_toggle.pack()

# Configuration du style pour le bouton transparent
style = ttk.Style()
style.configure("Transparent.TButton", background="systemTransparent", foreground="#ffffff", borderwidth=0)

# Lancement de la boucle principale
window.mainloop()
