# 环境准备
```
# bridge
git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
git checkout bc58ea9c
git apply mcore-bridge.bc58ea9c.patch
pip install -e .

# megatron
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.16.0
git apply megatron.0.16.0.patch

# mindpseed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout core_r0.16.0
git apply mindspeed.0.16.0.patch

# ms-swift
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout e672d99e
pip install -e .
```
其它依赖：
```
"transformers==5.16.1"
"flash-linear-attention[npu]==0.5.2"
ninja
"triton-ascend==3.2.1"
qwen-vl-utils==0.0.14
```

# 训练任务
多机任务，依次在机器上执行
`NODE_RANK=i bash train_sft.sh`