# 一、构建镜像CI

资源列表

| 资源类型 | runner名 | 是否可用 | 作用 |
|---|---|---|---|
| CPU |  ubuntu-22.04-arm | 是 |  构建镜像(pull基础镜像，push turbo镜像) |
| CPU |  linux-aarch64-cpu-x(4\24) | 否 | 构建镜像(pull基础镜像，push turbo镜像) |
| NPU |  linux-aarch64-a3-800i-x(2\4\6\8) | 是 | 测试用例 NPU计算 |


该部分使用CPU资源，分为三步：

1. pull cann基础镜像
2. build cann镜像为turbo镜像
3. push 镜像，以供后续PR CI使用

**pull image**

| 资源 | 阶段 | 网速 | 时间 |
|---|---|---|---|
|linux-aarch64-cpu-4 | FROM quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11|  <90Mb/s | 2min 12s |
| linux-aarch64-cpu-4| swr.cn-north-12.myhuaweicloud.com/ascend/torchturbo_npu:base-latest | <4Mb/s  | 14min |


**build image**

| 资源 | 阶段 | 耗时 | 占比 |
|---|---|---|---|
|linux-aarch64-cpu-4 | `#10` 编译 fla-npu（Ascend 算子） | **84.7 min** | 40.5% |
| linux-aarch64-cpu-4| `#12` 源码编译 PyTorch 2.10.0 | **68.1 min** | 32.5% |
| linux-aarch64-cpu-4| `#17` 导出并推送镜像到 SWR | **50.1 min** | 23.9% |
| linux-aarch64-cpu-4| 其余全部（拉基础镜像 + pip + 收尾） | 6.9 min | 3.1% |

**push image**

| 资源| 子阶段 | 耗时 |
|---|---|---|
| linux-aarch64-cpu-4| `exporting layers`（本地写盘） | 47.6 s |
| linux-aarch64-cpu-4 | **`pushing layers`（上传 SWR）** | **2951.6 s ≈ 49.2 min** |
| linux-aarch64-cpu-4 | push manifest（两个 tag） | 2.9 s + 0.5 s |


# 二、PR CI

该部分使用NPU pod，分为两步：

1. pull turbo镜像

2. 执行测试用例

**pull turbo镜像**

| 资源 | 阶段 | 网速 | 时间 |
|---|---|---|---|
|linux-aarch64-a3-800i-8 | FROM quay.io/ascend/cann:9.0.0-a3-openeuler24.03-py3.11|  | 42min |
| linux-aarch64-a3-800i-8| FROM 	swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-a3-openeuler24.03-py3.11| | 3min 12s  |

