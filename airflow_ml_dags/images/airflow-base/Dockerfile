FROM python:3.8-slim-buster

#link do sample https://pythonspeed.com/articles/activate-virtualenv-dockerfile/

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
