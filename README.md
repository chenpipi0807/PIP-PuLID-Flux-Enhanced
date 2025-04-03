# ComfyUI-PuLID-Flux-Enhanced

改编自 https://github.com/chenpipi0807/PIP-PuLID-Flux-Enhanced.git

部分思路借鉴：https://github.com/balazik/ComfyUI-PuLID-Flux

**最近更新**: 2025-04-03 - 添加InfiniteYou增强方法

![微信截图_20250402175311](https://github.com/user-attachments/assets/53586740-0563-48c5-b463-58688b3e5f30)
![微信截图_20250402181106](https://github.com/user-attachments/assets/2e3bd0e6-c7f7-4906-b254-07e87e6b1b05)
![微信截图_20250402174710](https://github.com/user-attachments/assets/f37d718c-a99c-42a4-a5e2-3743283c5970)


## 核心功能

### 高级身份保持和姿势解耦方法

本项目提供不同的方法来实现身份保持和姿势控制：

1. **保真度（Fidelity）**：经典方法，专注于身份保持，基于正交投影v2。

2. **自适应通道选择（Adaptive）** - 基于特征相关性分析动态识别哪些特征通道与姿势或身份相关。这种方法比固定通道策略提供更精确的姿势解耦，能够自动适应不同的模型和图像。

3. **残差连接方法（Residual）** - 实现了InfiniteYou方法的简化版本，直接添加身份特征而不是使用正交投影。这种方法通常产生更高保真度的结果，同时仍然保持良好的姿势变化。

4. **增强残差（Residual_Infu）** - PRO节点独有的方法，利用InfiniteYou的先进模型对残差方法进行增强，能够更好地处理复杂的表情和角度变化。

### 方法特点与选择指南

- **自适应方法**在需要保持提示和参考图像之间明显姿势差异时效果最佳。它特别适合于较强的引导比例。

- **残差方法**通常提供最高的身份保真度，同时仍然允许姿势变化。尝试在权重为1.5-2.5时使用此方法。

- **增强残差方法**使用先进的深度学习模型处理图像，在保持身份的同时提供更自然的姿势变化。推荐权重1.5-2.5。

## 解决的问题

本项目主要解决了AI图像生成中的以下问题：

1. **身份保持与姿势灵活性的平衡** - 传统方法在保持身份特征的同时往往会限制姿势变化，我们的方法提供了更好的平衡。

2. **特征空间更精确的分离** - 通过自适应通道选择和残差连接，更准确地区分身份和姿势特征，提高生成质量。

3. **更广泛的适用性** - 提供更好的文本遵从一定程度上降低畸形概率。

## 使用方法

### 安装

1. 将本仓库克隆到ComfyUI的`custom_nodes`目录下
2. (可选) 如需使用增强残差方法，请访问 https://huggingface.co/ByteDance/InfiniteYou/tree/main/infu_flux_v1.0 下载附加的INFU-models文件
   - 下载InfuseNet模型文件，包含相似度和美学模型
   - 将下载的文件放入`custom_nodes/ComfyUI-PuLID-Flux-Enhanced/INFU-models`文件夹
   - 确保文件结构为：`sim_stage1/image_proj_model.bin`和`aes_stage2/image_proj_model.bin`
3. 重启 ComfyUI
4. 调整权重参数（residual_infu和residual推荐1.5-2.5之间）来达到理想的效果

### 使用

1. 在ComfyUI中，搜索并添加 **Apply PIP-PuLID-Flux** 节点
2. 连接你的条件（提示）和参考图像到节点
3. 在节点设置中选择不同的方法：
   - **Fidelity**：当你需要最大程度保持参考图像的身份特征时选择
   - **Adaptive**：当你需要在身份保持和姿势灵活性之间取得平衡时选择
   - **Residual**：当你想要高保真度的身份特征，同时保持对姿势的控制时选择
   - **Residual_Infu**：当你需要在身份保持和姿势灵活性之间取得平衡时选择

4. 高级用户可以尝试 **Apply PIP-PuLID-Flux PRO** 节点，它提供了额外的 **Residual_Infu** 方法，要使用此方法需要额外的模型文件
  
## 双人pulid（进行中的研究）
我的思路是通过动态拆分token分别确定潜空间中的两张人脸，但不是很顺利，主要是相似度上不去，有时间再说吧。
感兴趣可以看一下pip_pulidflux_pro.py.bak
![微信截图_20250403112639](https://github.com/user-attachments/assets/2ad41784-5194-44b8-a525-ae12ed1fdef7)


## 目录结构：

INFU-models/
├── aes_stage2/
│   ├── image_proj_model.bin
│   └── InfuseNetModel/
│       ├── config.json
│       ├── diffusion_pytorch_model-00001-of-00002.safetensors
│       ├── diffusion_pytorch_model-00002-of-00002.safetensors
│       └── diffusion_pytorch_model.safetensors.index.json
└── sim_stage1/
    ├── image_proj_model.bin
    └── InfuseNetModel/
        ├── config.json
        ├── diffusion_pytorch_model-00001-of-00002.safetensors
        ├── diffusion_pytorch_model-00002-of-00002.safetensors
        └── diffusion_pytorch_model.safetensors.index.json
