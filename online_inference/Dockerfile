FROM python:3.8-slim
COPY docker_requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY model/model.pkl    \
     src/app.py         \
     src/entites.py     \
     ./

WORKDIR .

ENV PATH_TO_MODEL="/model.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
