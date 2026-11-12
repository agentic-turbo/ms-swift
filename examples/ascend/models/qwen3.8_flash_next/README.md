# Qwen3.8-Flash-Next Ascend SFT

本示例通过 ms-swift、mcore-bridge 和 MindSpeed，在 Ascend NPU 上进行 Qwen3.8-Flash-Next 全参数 SFT。配套补丁包含模型 NPU 适配和 PLE 表 EP 分片；默认脚本使用 8 机 × 16 卡，开启 BF16、数据 packing 和优化器 CPU offload。

## 环境准备

本任务已在基础镜像 `quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12` 上验证。请先准备与集群匹配的驱动、CANN、PyTorch 和 torch-npu，并确保各节点使用相同的软件环境。

```bash
python -m pip install \
    "transformers==5.16.1" \
    "triton-ascend==3.2.1" \
    "qwen-vl-utils==0.0.14" \
    ninja

python -m pip install "flash-linear-attention[npu]==0.5.2"
python -m pip uninstall -y triton triton-ascend
python -m pip install --no-cache-dir "triton-ascend==3.2.1"
```

按以下版本准备源码，并应用本目录中的三个补丁。`EXAMPLE_DIR` 指向当前示例目录的绝对路径，`WORK_DIR` 是存放各依赖仓库的目录。

```bash
export EXAMPLE_DIR=/path/to/ms-swift/examples/ascend/models/qwen3.8_flash_next
export WORK_DIR=/path/to/workspace
mkdir -p "$WORK_DIR"

# Megatron-LM
cd "$WORK_DIR"
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.16.0
git apply "$EXAMPLE_DIR/megatron.0.16.0.patch"

# MindSpeed
cd "$WORK_DIR"
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout core_r0.16.0
git apply "$EXAMPLE_DIR/mindspeed.0.16.0.patch"

# mcore-bridge
cd "$WORK_DIR"
git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
git checkout 8b567b34e3f9a1087624d45ef0e9474f1021a2f3
git apply "$EXAMPLE_DIR/mcore-bridge.8b567b34.patch"
python -m pip install -e .

# ms-swift
cd "$WORK_DIR"
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout e672d99e
python -m pip install -e .
```

已准备好对应源码时，可跳过 `git clone`，在各仓库中切换版本并应用补丁即可。本示例通过训练脚本中的 `PYTHONPATH` 加载 Megatron-LM 和 MindSpeed 源码，不额外执行 pip 安装。`mcore-bridge` 和 `ms-swift` 需要执行安装：前者提供包版本元数据，后者注册 `megatron` 命令，仅设置 `PYTHONPATH` 不能替代安装。

## 训练配置

启动前修改 [train_sft.sh](train_sft.sh) 中的以下内容：

| 配置 | 说明 |
| --- | --- |
| 环境初始化脚本、`PYTHONPATH` | 按实际 CANN 和源码安装路径修改。 |
| `NNODES`、`NPROC_PER_NODE`、`ASCEND_RT_VISIBLE_DEVICES` | 默认 8 个节点、每节点 16 卡，按实际资源调整。 |
| `MASTER_ADDR`、`MASTER_PORT` | 所有节点填写相同的主节点地址和端口。 |
| `GLOO_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME` | 填写各节点用于通信的网卡名称。 |
| `MODEL`、`DATASET`、`OUTPUT` | 分别填写模型权重、训练数据和输出目录，确保相关节点可访问。 |

默认并行配置为 TP=8、PP=2、CP=2、EP=64，专家 TP=1，并开启 sequence parallel。PLE 表按 EP 分片；调整节点数或卡数时，需要同步检查并行配置，不能只修改 `NNODES`。

训练默认冻结视觉编码器、对齐模块和 QSA indexer；最大序列长度为 4096，micro batch size 为 1，global batch size 为 32。`PLE_CPU_OFFLOAD=0` 保持 PLE 表参与训练，`--optimizer_cpu_offload true` 则将优化器状态卸载到 CPU，两者控制不同内容。

脚本默认训练 100 步。需要启用 MTP 时，在 `megatron sft` 参数中加入：

```bash
--mtp_num_layers 1 \
--mtp_loss_scaling_factor 0.1 \
```

## 启动训练

在每个节点进入本示例目录并执行以下命令，`RANK` 是从 0 开始的节点编号。默认 8 机任务需在全部节点启动，编号为 0～7：

```bash
# 主节点
RANK=0 bash train_sft.sh

# 第二个节点，其余节点依次使用 RANK=2 ... RANK=7
RANK=1 bash train_sft.sh
```

脚本会将 `RANK` 赋给 `NODE_RANK`，并通过 `megatron sft` 启动分布式训练。日志和权重保存在 `OUTPUT` 指定目录；默认不保存优化器和随机数状态。若需要在 100 步内按步保存权重，请相应调小默认的 `--save_steps 500`。
