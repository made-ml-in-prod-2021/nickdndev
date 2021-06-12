FROM airflow-base

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY predict.py /predict.py

ENTRYPOINT ["python", "predict.py"]
