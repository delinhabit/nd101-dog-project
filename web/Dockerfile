FROM ubuntu:16.04

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN pip install -U pip

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

CMD ["python", "src/app.py"]
