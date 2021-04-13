from concurrent import futures
import grpc
from service_pb2 import HandwrittenNumberResponse 
import service_pb2_grpc
import joblib
from PIL import Image
import numpy as np
from skimage.transform import rescale, resize
from grpc_interceptor import ServerInterceptor
from grpc_interceptor import ExceptionToStatusInterceptor
import logging
from datetime import datetime
from signal import signal, SIGTERM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load("model.pkl")

def standarize(array: np.ndarray) -> np.ndarray:
    array_min = array.min()
    array_max = array.max()
    new_array = 255*(array-array_min)/(array_max-array_min)
    return new_array

class GeneralLogger(ServerInterceptor):
     def intercept(self, method, request, context, method_name):
         try:
             result = method(request, context)
             today = str(datetime.today())
             logger.info(f"{today} {method_name} result {result[1][1]} probability {result[1][0]}")
             return result[0]
         except Exception as e:
             self.log_error(e)
             raise

     def log_error(self, e: Exception):
         logger.warn(f"{e} has occured")
        

class PredictionsService(service_pb2_grpc.PredictionsServicer):
     def PredictNumber(self, request, context):
         data = request.image_bytes
         data = np.array([int(pixel) for pixel in data.decode().split("-")])
         data = data.reshape((280, 280))
         data = rescale(data,0.1, anti_aliasing=False).reshape((28*28))
         data = standarize(data)
         result = model.predict(np.array([data]))[0]
         probability = model.predict_proba(np.array([data]))[0]
         probability = probability[0], probability.argmax()
         return HandwrittenNumberResponse(image_number=int(result)), probability 

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
           interceptors = [GeneralLogger(),ExceptionToStatusInterceptor()])
    service_pb2_grpc.add_PredictionsServicer_to_server(
           PredictionsService(), server)
   
    with open("server.key", "rb") as fp:
        server_key = fp.read()

    with open("server.pem", "rb") as fp:
        server_cert = fp.read()

    with open("ca.pem","rb") as fp:
        ca_cert = fp.read()

    creds = grpc.ssl_server_credentials(
                    [(server_key, server_cert)],
                    root_certificates=ca_cert,
                    require_client_auth=True)

    server.add_secure_port("[::]:443", creds)

   
    server.start()

    def handle_sigterm(*_):
        print("Received shutdown signal")
        all_rpcs_done_event = server.stop(5)
        all_rpcs_done_event.wait(5)
        print("Shut down gracefully")

    
    signal(SIGTERM, handle_sigterm)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
