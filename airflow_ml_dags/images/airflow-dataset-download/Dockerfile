FROM airflow-base

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY download.py /download.py

ENTRYPOINT ["python", "download.py"]
