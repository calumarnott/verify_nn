#!/bin/bash

 vehicle verify \
  --specification drone.vcl \
  --verifier Marabou \
  --network droneNN:clean_model-20251026-214014.onnx \
  --property property3