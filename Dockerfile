# syntax=docker/dockerfile:1
FROM python:3.8-slim
EXPOSE 4040
WORKDIR /churn-prediction
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD python app.py