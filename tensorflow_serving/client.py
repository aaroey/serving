def mytest(server='0.0.0.0:8500'):
  import numpy as np
  import grpc
  import tensorflow as tf
  from tensorflow_serving.apis import predict_pb2
  from tensorflow_serving.apis import prediction_service_pb2_grpc
  channel = grpc.insecure_channel(server)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  input_shape = [128, 224, 224, 3]
  for _ in range(10):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'mymodel'
    request.model_spec.signature_name = 'predict'
    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(
            # Synthetic inputs.
            np.random.uniform(input_shape).astype(np.float32),
            shape=input_shape))
    response = stub.Predict(request)
    predicted_label = np.array(response.outputs['classes'].int64_val)
    print('prediction result: ' + str(predicted_label))
