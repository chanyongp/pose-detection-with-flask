FROM python:3.8.0

RUN apt-get update && apt-get install -y sqlite3 && apt-get install -y libsqlite3-dev

WORKDIR /usr/src/

COPY ./apps /usr/src/apps
COPY ./local.sqlite /usr/src/local.sqlite
COPY ./requirements.txt /usr/src/requirements.txt
COPY ./model.pt /usr/src/model.pt

RUN pip install --upgrade pip

RUN pip install torch torchvision opencv-python

RUN pip install -r requirements.txt

RUN echo "building..."

ENV FLASK_APP "apps.app:create_app"
ENV IMAGE_URL "/storage/images/"

EXPOSE 5000

CMD ["flask", "run", "-h", "0.0.0.0"]