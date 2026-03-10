
Start Docker Desktop and go to the folder, open terminal and run

docker build -t nn-meter-env .

If PyCharm Professional
File → Settings → Project → Python Interpreter

If PyCharm Community + Terminal
docker run -it -v %cd%:/app nn-meter-venv
docker run -it -v %cd%:/app nn-meter-venv python main.py