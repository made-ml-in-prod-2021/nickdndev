FROM python:3.6-stretch
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY model.pkl /model.pkl
COPY src/app.py src/app.py

WORKDIR /src

ENV PATH_TO_MODEL="/model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
