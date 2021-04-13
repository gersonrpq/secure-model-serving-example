# syntax = docker/dockerfile:1.0-experimental
FROM python

WORKDIR /app

COPY ./protos/service.proto .

COPY ./models-server/requirements.txt .

COPY ./ca.pem . 

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

COPY ./models-server .

RUN python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. service.proto

RUN openssl req -nodes -newkey rsa:4096 -subj /CN=model-server \
                -keyout server.key -out server.csr

RUN --mount=type=secret,id=ca.key \
    openssl x509 -req -in server.csr -CA ca.pem -CAkey /run/secrets/ca.key \
                 -set_serial 1 -out server.pem

EXPOSE 443

ENTRYPOINT [ "python", "main.py" ]
