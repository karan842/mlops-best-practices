# syntax=docker/dockerfile:1
FROM python:3.8
EXPOSE 4000
WORKDIR /customer-churn
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD python app.py