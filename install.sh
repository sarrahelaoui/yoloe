#!/bin/bash
# Installation automatique — VISIONBENCH (YOLOE visual-prompt console)
# Usage: bash install.sh

set -e

echo "== 1/4 : Création de l'environnement virtuel =="
python3 -m venv venv
source venv/bin/activate

echo "== 2/4 : Mise à jour de pip =="
pip install --upgrade pip

echo "== 3/4 : Installation des dépendances =="
pip install -r backend/requirements.txt

echo "== 4/4 : Téléchargement du modèle YOLOE (au premier lancement) =="
python3 -c "from ultralytics import YOLOE; YOLOE('yoloe-11l-seg.pt')"

echo ""
echo "Installation terminée."
echo "Pour lancer le backend :"
echo "  source venv/bin/activate"
echo "  cd backend && python app.py"
echo ""
echo "Puis ouvre frontend/index.html dans ton navigateur."
