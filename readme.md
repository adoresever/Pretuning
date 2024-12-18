# Pretuning

[English](#english) | [中文](#chinese)

<a name="english"></a>
## 🌟 Pretuning

A tool designed for creating pre-training datasets for language models, supporting one-click batch processing of both text and image datasets.

### 🚀 Features

- 📝 Text Dataset Creation
  - Smart text processing using pydanticai agents
  - Automatic text segmentation
  - Batch text processing
  - Standard instruction format output

- 🖼️ Image Dataset Creation
  - Automatic image preprocessing
  - Batch image description generation
  - Custom prompt support
  - OpenAI API integration

### 🔧 Quick Start

1. Clone repository:
```bash
git clone https://github.com/adoresever/pretuning.git
cd pretuning
```

2. Create and activate conda environment:
```bash
conda create -n pretuning python=3.10 -y
conda activate pretuning
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start application:
```bash
python main.py
```

5. Access the web interface (typically http://localhost:7860)

### 📦 Project Structure
```
Pretuning/
├── config/            # Configuration files
├── core/             # Core functionality modules
├── ui/               # Web interface
├── input/            # Example input files
├── image_dataset/    # Image dataset output
├── text_dataset/     # Text dataset output
└── temp_dataset/     # Temporary files
```

### 📝 Usage

#### For Text Dataset:
1. Upload text files
2. Set processing parameters in the UI
3. Generate training data with one click
4. Export in standard format

#### For Image Dataset:
1. Upload image files
2. Configure API settings in the UI
3. Generate batch descriptions
4. Export dataset

### 🛠️ Requirements

- Python 3.8+
- Anaconda or Miniconda
- Sufficient disk space for dataset storage

---

<a name="chinese"></a>
## 🌟 预训练数据集制作工具

一个专为语言模型预训练设计的数据集制作工具，支持文本和图像数据集的一键式批量处理。

### 🚀 功能特点

- 📝 文本数据集制作
  - 使用 PydanticAi agents 进行智能文本处理
  - 自动文本分段
  - 批量文本处理
  - 标准指令格式输出

- 🖼️ 图像数据集制作
  - 自动图像预处理
  - 批量图像描述生成
  - 支持自定义提示词
  - OpenAI API 集成

### 🔧 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/adoresever/pretuning.git
cd pretuning
```

2. 创建并激活 conda 虚拟环境：
```bash
conda create -n pretuning python=3.10 -y
conda activate pretuning
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 启动应用：
```bash
python main.py
```

5. 访问Web界面（通常是 http://localhost:7860）

### 📦 项目结构
```
Pretuning/
├── config/            # 配置文件
├── core/             # 核心功能模块
├── ui/               # Web界面
├── input/            # 示例输入文件
├── image_dataset/    # 图像数据集输出
├── text_dataset/     # 文本数据集输出
└── temp_dataset/     # 临时文件
```

### 📝 使用方法

#### 文本数据集：
1. 上传文本文件
2. 在界面中设置处理参数
3. 一键生成训练数据
4. 导出标准格式

#### 图像数据集：
1. 上传图片文件
2. 在界面中配置 API 设置
3. 批量生成描述
4. 导出数据集

### 🛠️ 环境要求

- Python 3.8+
- Anaconda 或 Miniconda
- 足够的磁盘空间用于数据集存储

## 👥 作者

- **王宇** (Wang Yu) - [Wywelljob@gmail.com](mailto:wywelljob@gmail.com)

## 📄 开源协议

本项目采用 MIT 协议 - 详见 [LICENSE](LICENSE) 文件
