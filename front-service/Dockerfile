# syntax = docker/dockerfile:1.0-experimental
FROM python:3.8

WORKDIR /app

COPY ./protos/service.proto .

COPY ./front-service/requirements.txt .

COPY ./ca.pem . 

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY ./front-service .

RUN python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. service.proto

RUN openssl req -nodes -newkey rsa:4096 -subj /CN=front-service \
                -keyout client.key -out client.csr

RUN --mount=type=secret,id=ca.key \
    openssl x509 -req -in client.csr -CA ca.pem -CAkey /run/secrets/ca.key \
                 -set_serial 1 -out client.pem

EXPOSE 8501

ENTRYPOINT ["streamlit","run","--server.address","0.0.0.0","main.py"]


