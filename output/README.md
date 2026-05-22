# superpoint_v1_sim_int32.onnx
    cd python
    python node_modify.py
    /usr/src/tensorrt/bin/trtexec --onnx=superpoint_v1_sim_int32.onnx --saveEngine=superpoint_v1_sim_int32.engine --minShapes=input:1x1x100x100  --optShapes=input:1x1x500x500 --maxShapes=input:1x1x1500x1500

# plnet_s0.onnx

    /usr/src/tensorrt/bin/trtexec --onnx=plnet_s0.onnx --saveEngine=plnet_s0.engine --minShapes=input:1x1x100x100  --optShapes=input:1x1x512x512 --maxShapes=input:1x1x1500x1500

# plnet_s1.onnx

  /usr/src/tensorrt/bin/trtexec \
  --onnx=plnet_s1.onnx \
  --saveEngine=plnet_s1.engine \
  --fp16 \
  --minShapes=juncs_pred:1x2,lines_pred:1x4,idx_lines_for_junctions:1x2,inverse:1x1,iskeep_index:1x1,loi_features:1x16x16x16,loi_features_thin:1x4x16x16,loi_features_aux:1x4x16x16 \
  --optShapes=juncs_pred:250x2,lines_pred:20000x4,idx_lines_for_junctions:20000x2,inverse:20000x1,iskeep_index:20000x1,loi_features:1x128x128x128,loi_features_thin:1x4x128x128,loi_features_aux:1x4x128x128 \
  --maxShapes=juncs_pred:500x2,lines_pred:50000x4,idx_lines_for_junctions:40000x2,inverse:40000x1,iskeep_index:40000x1,loi_features:1x256x512x512,loi_features_thin:1x4x512x512,loi_features_aux:1x4x512x512

