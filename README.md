# UITARS Mobile Agent - 移动端智能体操作平台

本项目实现了一个基于 **UITARS** 大模型和 **GUI Agent** 技术的移动端智能体，能够通过 ADB 指令智能操控 Android 手机。项目集成了 FastAPI 后端服务与 React 可视化前端，支持实时屏幕投射、自然语言指令交互及操作过程的可视化监控。

## ✨ 主要功能

- **🤖 智能操控**: 能够理解自然语言指令（如“打开设置并开启蓝牙”），自动规划并执行一系列手机操作。
- **📱 实时投屏**: 前端界面实时显示手机当前屏幕画面，低延迟、无缝刷新。
- **🧠 思维链展示**: 可视化展示 Agent 的思考过程（Thought）和执行动作（Action），便于调试和观察。
- **⚡ 高效执行**: 优化的截图与指令执行逻辑，操作响应迅速。

  ![1765361999065](image/README/1765361999065.png)

https://github.com/user-attachments/assets/3380a514-8e6a-48c7-ab00-2e68af41d055

## ✨ 12.20更新：使用服务器vllm部署qwen3vl替代uitars
### 1.远程服务器连接手机ADB
在连接主机开始ADB调试后，打开无线调试选项（会显示ip:port），之后ssh输入adb connect [ip]:[port]，同时在手机上确认配对，即可远程调试ADB
### 2.vllm部署qwen3vl
```bash
conda create -n vllm python = 3.10
conda activate vllm
pip install vllm
mkdir vllm_deploy
cd vllm_deploy
```
下载模型
```bash
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir Qwen/Qwen3-VL-8B-Instruct
vllm serve Qwen/Qwen3-VL-8B-Instruct/ --trust-remote-code --tensor-parallel-size 2 --max-model-len 65536  --port 8000 --host 0.0.0.0 --dtype bfloat16
```
修改代码逻辑适配qwen3vl
### 3.llamafactory微调Qwen3vl-8B
在使用原始模型是，发现器在处理GUI情况时有几个问题：游戏能力表现弱（消消乐用点而不是滑）无法识别程序中间状态（抖音后台若处于聊天界面，模型打开抖音会说这是聊天程序，之后退出）
针对这几个问题，构建STF数据集（其实做强化学习更好）进行LoRA微调
模型效果有明显提升


## 📂 项目结构

```text
UITARS_MobileAgent_V1/
├── backend/                  # 后端核心代码
│   ├── MobileAgent/          # Agent 核心逻辑库 (图像处理、Prompt构建等)
│   ├── codes/                # 工具类代码
│   ├── service/              # FastAPI 服务端入口 (main.py, agent_runner.py)
│   ├── tools/                # 辅助工具 (如 ADBKeyboard)
│   └── run_uitars.py         # 命令行运行脚本 (备用)
├── frontend/                 # React 前端代码
│   ├── src/                  # 源代码
│   └── ...
├── .gitignore                # Git 忽略配置
├── requirements.txt          # 环境依赖 
└── README.md                 # 项目说明文档
```

## 🛠️ 环境要求

- **Python**: 3.10 或更高版本
- **Node.js**: 16 或更高版本
- **ADB (Android Debug Bridge)**: 需安装并添加到系统环境变量 PATH 中
- **Android 设备**: 一台开启了“开发者模式”和“USB调试”的 Android 手机或模拟器

## 🚀 安装指南

### 1. 后端环境配置

在项目根目录下执行：

```bash
# 推荐使用 conda 创建虚拟环境
conda create -n uitars python=3.10
conda activate uitars

# 安装依赖
pip install -r requirements.txt
```

> **注意**: 请确保您的环境中正确安装了 PyTorch 及其他相关深度学习库。

### 2. 前端环境配置

进入前端目录并安装依赖：

```bash
cd frontend
npm install
```

### 3. ADB与手机adb调试配置

windows的adb可以网上找教程下载并添加到系统变量中

打开cmd，输入adb devices即可查看当前连接的移动设备，下面说下如何配置手机的adb环境

1.开启开发者模式

设置->关于手机->版本号连点7-8次->显示你已进入开发者模式

2.打开USB调试

设置->系统->开发人员选项->打开USB调试->等待出现弹窗，点击信任此电脑

3.安装adbkeyboard实现打字输入

打开cmd，在backend/tools目录下，输入adb install ADBKeyboard.apk

设置->输入法->开启adbkeyboard输入法->重启

## 🖥️ 使用说明

### 第一步：启动后端服务

请确保手机已连接电脑，并在终端输入 `adb devices` 能看到设备。

在项目根目录下运行：

```bash
python backend/service/main.py
```

后端服务将在 `http://localhost:8000` 启动。

### 第二步：启动前端界面

在 `frontend` 目录下运行：

```bash
npm run dev
```

启动后，浏览器通常会自动打开 `http://localhost:5173`。

### 第三步：开始使用

1. 在网页界面的输入框中输入您的指令（例如：“打开网易云音乐播放每日推荐”）。
2. 点击 **Start Agent** 按钮。
3. 观察左侧的手机实时画面和右侧的 Agent 运行日志。

## ⚙️ 配置说明

如果您的 ADB 未添加到系统环境变量，或者需要修改模型配置，请编辑 `backend/service/agent_runner.py` 文件：

```python
# 修改 ADB 路径
self.adb_path = os.getenv("ADB_PATH", "C:\\your\\path\\to\\adb.exe")

# 修改模型配置 (如 API Key 等)
self.token_uitars = "your-api-token"
```

字节的uitars模型api每个新用户有免费额度，点击[火山方舟管理控制台](https://console.volcengine.com/ark/region:ark+cn-beijing/model?vendor=Bytedance&view=DEFAULT_VIEW)下拉找到Doubao-1.5-UI-TARS模型，点击立即体验之后，

## 📄 许可证

[MIT License]
