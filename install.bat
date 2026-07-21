@echo off
ECHO == 1/4 : Creation de l'environnement virtuel ==
python -m venv venv
CALL venv\Scripts\activate.bat

ECHO == 2/4 : Mise a jour de pip ==
python -m pip install --upgrade pip

ECHO == 3/4 : Installation des dependances ==
pip install -r backend\requirements.txt

ECHO == 4/4 : Telechargement du modele YOLOE ==
python -c "from ultralytics import YOLOE; YOLOE('yoloe-11l-seg.pt')"

ECHO.
ECHO Installation terminee.
ECHO Pour lancer le backend :
ECHO   venv\Scripts\activate
ECHO   cd backend ^&^& python app.py
ECHO.
ECHO Puis ouvre frontend\index.html dans ton navigateur.
PAUSE
