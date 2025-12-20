import os
import shutil
import re
import logging
import sys
import requests
import json
import base64

from logging.handlers import RotatingFileHandler
from PIL import Image
# 移除UITARS相关导入，保留MobileAgent核心功能
from MobileAgent.controller import get_screenshot, type, execute_action
from MobileAgent.chat import init_action_chat_uitars, add_response_uitars, add_box_token
from codes.utils import parse_action_to_structure_output,parsing_response_to_pyautogui_code,convert_coordinates

####################################### 修改后的配置 #########################################
# Your ADB path（保留原有ADB配置）
adb_path = os.getenv("ADB_PATH")
if not adb_path:
    if shutil.which("adb"):
        adb_path = "adb"
    else:
        # Fallback for Windows if not in PATH
        adb_path = "C:\\adb\\platform-tools\\adb"

adb_path = f'"{adb_path}"' if " " in adb_path and not adb_path.startswith('"') else adb_path

# Your instruction（保留原有指令）
instruction = "打开抖音极速版，一直刷抖音，用的动作是down"

# 【核心修改】vLLM部署的Qwen3-VL配置
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"  # vllm服务地址+端口
VLLM_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct-LoRA/"  # 和启动vllm时的模型名一致
VLLM_MAX_TOKENS = 2048  # 生成最大长度
VLLM_TEMPERATURE = 0.7  # 采样温度

#设置可以查看最近的多少步的信息（保留原有逻辑）
history_n = 7

###################################################################################################

# 【新增】替换原UITARS的encode_image函数，适配vLLM
def encode_image(image_path):
    """将图片转为Base64编码（vLLM兼容格式）"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 【核心修改】替换原inference_chat_uitars函数，调用本地vLLM API
def inference_chat_uitars(chat_action, model_name, api_url, token):
    """
    调用本地vLLM部署的Qwen3-VL进行推理
    参数说明：
    - chat_action: 原有构造的对话指令（保留）
    - model_name/api_url/token: 兼容原有参数名，实际使用VLLM配置
    """
    # 构造vLLM的请求消息（多模态格式）
    messages = []
    # 第一步：添加系统指令（确保模型按指定格式输出Thought和Action）
    system_prompt = """
    你是一个移动端UI交互代理，需要根据截图和指令执行相应操作。
    输出格式必须严格遵循：
    Thought: 你的思考过程
    Action: 具体操作（支持click/type/finished/wait/drag等，格式如click(start_box='(100,200))或drag(start_box='(100,200)', end_box='(300,400)')）。
    注意：坐标必须是整数或浮点数，不能是自然语言描述。
    """.strip()
    messages.append({"role": "system", "content": system_prompt})
    
    # 第二步：解析原有chat_action，提取用户指令和图片信息
    # 从chat_action中提取核心指令（原有逻辑）
    user_content = []
    # 1. 添加文本指令
    user_content.append({"type": "text", "text": chat_action})
    # 2. 添加历史截图（如果有）- 从全局history_images中获取最新的
    if history_images:
        latest_image_base64 = history_images[-1]
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{latest_image_base64}"}
        })
    
    messages.append({"role": "user", "content": user_content})

    # 构造vLLM请求体
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": VLLM_MAX_TOKENS,
        "temperature": VLLM_TEMPERATURE,
        "stream": False  # 非流式输出，便于解析
    }

    try:
        # 调用vLLM API
        response = requests.post(
            VLLM_API_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=60  # 超时时间
        )
        response.raise_for_status()  # 抛出HTTP错误
        
        # 解析响应
        result = response.json()
        output = result["choices"][0]["message"]["content"].strip()
        logger.info(f"vLLM推理结果：\n{output}")
        return output
    
    except Exception as e:
        logger.error(f"调用vLLM失败：{str(e)}", exc_info=True)
        return "Thought: 调用模型失败\nAction: wait()"

def get_perception_infos(adb_path, screenshot_file):
    """保留原有感知信息获取逻辑"""
    get_screenshot(adb_path)
    width, height = Image.open(screenshot_file).size
    return width, height

##配置日志记录器（保留原有配置）
logger = logging.getLogger('Qwen3-VL')
logger.setLevel(logging.INFO)

file_handler = RotatingFileHandler(
    'qwen3_vl.log',
    maxBytes=1024*1024*10,
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
#############

##历史思考和动作list初始化（保留原有逻辑）
thoughts = []
actions = []
history_images = []
history_responses = []

# max_pixels = 16384 * 28 * 28
max_pixels = 1280 * 28 * 28
min_pixels = 100 * 28 * 28

action = ""
temp_file = "temp"
screenshot = "screenshot"
if not os.path.exists(temp_file):
    os.mkdir(temp_file)
else:
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)
if not os.path.exists(screenshot):
    os.mkdir(screenshot)
error_flag = False
######################

##程序开始运行（核心逻辑保留，仅适配模型版本）
iter = 0
while True:
    iter += 1
    if iter == 1:
        screenshot_file = "./screenshot/screenshot.png"
        width, height = get_perception_infos(adb_path, screenshot_file)
        shutil.rmtree(temp_file)
        os.mkdir(temp_file)

    ##构建图片历史（保留原有逻辑）
    base64_image = encode_image(screenshot_file)
    history_images.append(base64_image)

    if len(history_images) > history_n:
        history_images = history_images[-history_n:]

    #####################

    ##构建模型输入（简化UITARS兼容逻辑）
    chat_action_init = init_action_chat_uitars(instruction)
    # 保留原有消息构造逻辑，但实际推理时会重新构造vLLM格式
    chat_action = add_response_uitars(chat_action_init, [])
    ##################

    ##【核心修改】调用vLLM推理（参数兼容原有格式，实际使用VLLM配置）
    output_action = inference_chat_uitars(
        chat_action, 
        VLLM_MODEL_NAME, 
        VLLM_API_URL, 
        ""  # vLLM无需token，传空值兼容原有参数
    )
    history_responses.append(output_action)

    ##解析Thought和Action（保留原有正则逻辑）
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:)", output_action, re.DOTALL)
    action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", output_action, re.DOTALL)

    thought = thought_match.group(1).strip() if thought_match else ""
    thoughts.append(thought)
    action_pre = action_match.group(1).strip() if action_match else ""
    
    if action_pre == "":
        bare_match = re.search(r"(click|left_single|left_double|right_single|hover|drag|scroll|type|press_home|press_back|finished|open_app|wait)\s*\([^\)]*\)", output_action, re.DOTALL)
        if bare_match:
            action_pre = bare_match.group(0).strip()

    # 【简化】移除UITARS版本判断，直接使用qwen25vl（Qwen3-VL兼容）
    model_type = "qwen3vl"

    ##动作预处理（保留原有逻辑）
    action_pre = re.sub(r"<\|box_start\|>|<\|box_end\|>", "", action_pre)

    ##动作解析和执行（保留原有逻辑）
    mock_response_dict = parse_action_to_structure_output(action_pre, 1000, height, width, model_type)
    parsed_pyautogui_code = parsing_response_to_pyautogui_code(mock_response_dict, height, width)
    action = convert_coordinates(mock_response_dict, height, width, model_type=model_type)
    actions.append(action)

    ##执行动作（保留原有逻辑）
    stop_flag = execute_action(action, adb_path)
    if stop_flag == "STOP":
        break
    ########

    ##截图更新（保留原有逻辑）
    last_screenshot_file = "./screenshot/last_screenshot.png"
    if os.path.exists(last_screenshot_file):
        os.remove(last_screenshot_file)
    os.rename(screenshot_file, last_screenshot_file)
    
    width, height = get_perception_infos(adb_path, screenshot_file)
    shutil.rmtree(temp_file)
    os.mkdir(temp_file)

    os.remove(last_screenshot_file)
