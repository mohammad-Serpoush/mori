
FROM python:3.12.7-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH .

WORKDIR /app/


RUN apt-get update
RUN apt-get install -y curl


COPY . /app/
RUN --mount=type=cache,target=/root/.cache pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache pip install wheel
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt

COPY ./runserver.sh /app/

RUN chmod +x /app/runserver.sh