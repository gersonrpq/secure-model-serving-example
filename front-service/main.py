import streamlit as st
from streamlit_drawable_canvas import st_canvas
import grpc
from service_pb2_grpc import PredictionsStub 
from service_pb2 import HandwrittenNumberRequest
import os
import numpy as np

st.set_page_config(
        page_title = "Handwritten Numbers Classifier",
        page_icon = ":pencil:",
)

def run():
    st.title("Write a number")

    canvas_result = st_canvas(
            stroke_width = 25,
            stroke_color = "#fff",
            background_color = "#000",
            height = 280,
            width = 280,
            drawing_mode = "freedraw",
            key = "canvas",
            )

    if st.button("Predict"):
        if canvas_result.image_data[:,:,:3].sum() > 0:
            output = number_prediction(canvas_result.image_data[:,:,0])
            st.success(f'The output is {output}')
        else:
            st.error("Please draw a number")


if __name__ == '__main__':
    
    SERVER_HOST = os.environ.get("SERVER_HOST", "[::]:9000")
    
    if SERVER_HOST:
        with open("client.key","rb") as fp:
            client_key = fp.read()
        with open("client.pem", "rb") as fp:
            client_cert = fp.read()
        with open("ca.pem", "rb") as fp:
            ca_cert = fp.read()
        creds = grpc.ssl_channel_credentials(ca_cert, client_key, client_cert)
        channel = grpc.secure_channel(SERVER_HOST, creds)
        client = PredictionsStub(channel)

    def number_prediction(image_array: np.ndarray ) -> int:
        
        image_bytes = '-'.join([str(pixel) for pixel in image_array.reshape((280*280))]).encode()
        request = HandwrittenNumberRequest(image_bytes = image_bytes)
        response = client.PredictNumber(request)

        return response.image_number
    run()
