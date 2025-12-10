import os
import shutil
import re
import logging
import sys
import threading
import time
import requests
from logging.handlers import RotatingFileHandler
from PIL import Image

# Add parent directory to sys.path to allow imports from MobileAgent and codes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from MobileAgent.api import encode_image
from MobileAgent.controller import get_screenshot, execute_action
from MobileAgent.chat import init_action_chat_uitars, add_response_uitars, add_box_token
from codes.utils import parse_action_to_structure_output, parsing_response_to_pyautogui_code, convert_coordinates

class UITARSRunner:
    def __init__(self):
        self.running = False
        self.instruction = ""
        self.adb_path = os.getenv("ADB_PATH", "C:\\adb\\platform-tools\\adb")
        self.adb_path = f'"{self.adb_path}"' if " " in self.adb_path and not self.adb_path.startswith('"') else self.adb_path
        
        self.uitars_version = "1.5"
        self.model_name = 'doubao-1-5-ui-tars-250428'
        self.API_url_uitars = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.token_uitars = "ENTER-YOUR-API-HERE"
        self.history_n = 5
        
        # State
        self.latest_log = ""
        self.latest_thought = ""
        self.latest_action = ""
        self.history_images = []
        self.history_responses = []
        self.thoughts = []
        self.actions = []
        self.iter = 0
        
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.screenshot_dir = os.path.join(self.base_dir, "screenshot")
        self.temp_dir = os.path.join(self.base_dir, "temp")
        self.screenshot_file = os.path.join(self.screenshot_dir, "screenshot.png")
        
        # Logger setup
        self.logger = logging.getLogger('UITARS_Backend')
        self.logger.setLevel(logging.INFO)
        # Avoid adding handlers multiple times if re-instantiated
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def update_instruction(self, new_instruction):
        self.instruction = new_instruction
        self.logger.info(f"Instruction updated to: {self.instruction}")

    def get_perception_infos(self):
        get_screenshot(self.adb_path)
        # Move screenshot to expected location if get_screenshot doesn't put it there directly
        # get_screenshot implementation in controller.py usually saves to ./screenshot/screenshot.png relative to cwd
        # We need to ensure it's in self.screenshot_file
        
        # Since get_screenshot might rely on CWD, let's check where it saves. 
        # Assuming it saves to ./screenshot/screenshot.png relative to current working directory.
        # We will handle CWD in the run loop.
        
        if os.path.exists(self.screenshot_file):
            width, height = Image.open(self.screenshot_file).size
            return width, height
        return 0, 0

    def _inference_chat_uitars_safe(self, chat, model, api_url, token):
        headers = {
            "Content-Type": "application/json",
            'Accept': 'application/json',
            "Authorization": f"Bearer {token}"
        }

        data = {
            "model": model,
            "messages": [],
            "max_tokens": 2048,
            'temperature': 0.0,
            "seed": 1234
        }

        for message in chat:
            data["messages"].append({
                "role": message["role"],
                "content": message["content"]
            })

        try:
            res = requests.post(api_url, headers=headers, json=data, timeout=60)
            if res.status_code != 200:
                return f"API Error {res.status_code}: {res.text}"
            
            res_json = res.json()
            if 'choices' in res_json and len(res_json['choices']) > 0:
                return res_json['choices'][0]['message']['content']
            else:
                return f"Unexpected response format: {res_json}"
        except Exception as e:
            return f"Network Error: {str(e)}"

    def run_loop(self):
        self.running = True
        self.logger.info("Agent loop started")
        
        # Reset state for new run
        self.iter = 0
        self.history_images = []
        self.history_responses = []
        self.thoughts = []
        self.actions = []
        
        # Ensure directories exist
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)
        else:
            shutil.rmtree(self.temp_dir)
            os.mkdir(self.temp_dir)
            
        if not os.path.exists(self.screenshot_dir):
            os.mkdir(self.screenshot_dir)

        # Check instruction
        if not self.instruction:
            self.logger.warning("No instruction provided. Waiting for instruction...")
            self.running = False
            return

        # Set CWD to base_dir so that relative paths in MobileAgent work as expected
        original_cwd = os.getcwd()
        os.chdir(self.base_dir)

        try:
            while self.running:
                self.iter += 1
                if self.iter == 1:
                    width, height = self.get_perception_infos()
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
                    os.mkdir(self.temp_dir)
                # else:
                #    # Refresh screenshot info
                #     width, height = self.get_perception_infos()

                # Build messages
                base64_image = encode_image(self.screenshot_file)
                self.history_images.append(base64_image)
                
                if len(self.history_images) > self.history_n:
                    self.history_images = self.history_images[-self.history_n:]

                messages, images = [], []
                for image in self.history_images:
                    images.append(image)

                image_num = 0
                if len(self.history_responses) > 0:
                    for history_idx, history_response in enumerate(self.history_responses):
                        if history_idx + self.history_n > len(self.history_responses):
                            encoded_string = images[image_num]
                            messages.append({
                                "role": "user",
                                "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                            })
                            image_num += 1
                            messages.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": add_box_token(history_response)}]
                            })

                    encoded_string = images[image_num]
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                    })
                    image_num += 1
                else:
                    encoded_string = images[image_num]
                    messages.append({
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}}]
                    })
                    image_num += 1

                # Inference
                chat_action_init = init_action_chat_uitars(self.instruction)
                chat_action = add_response_uitars(chat_action_init, messages)
                
                self.logger.info("Sending request to model...")
                output_action = self._inference_chat_uitars_safe(chat_action, self.model_name, self.API_url_uitars, self.token_uitars)
                self.history_responses.append(output_action)
                self.latest_log = output_action

                # Check for error
                if output_action.startswith("API Error") or output_action.startswith("Network Error") or output_action.startswith("Unexpected response"):
                    self.logger.error(f"Inference failed: {output_action}")
                    # time.sleep(5)
                    # continue
                    self.running = False
                    break

                # Parse output
                thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:)", output_action, re.DOTALL)
                action_match = re.search(r"Action:\s*(.*?)(?=\n|$)", output_action, re.DOTALL)

                thought = thought_match.group(1).strip() if thought_match else ""
                self.thoughts.append(thought)
                self.latest_thought = thought
                
                action_pre = action_match.group(1).strip() if action_match else ""
                if action_pre == "":
                    bare_match = re.search(r"(click|left_single|left_double|right_single|hover|drag|scroll|type|press_home|press_back|finished|open_app|wait)\s*\([^\)]*\)", output_action, re.DOTALL)
                    if bare_match:
                        action_pre = bare_match.group(0).strip()

                if self.uitars_version == "1.5":
                    model_type = "qwen25vl"
                elif self.uitars_version == "1.0":
                    model_type = "qwen2vl"
                else:
                    self.logger.error(f"uitars_version:{self.uitars_version} is not supported.")
                    model_type = "qwen2vl" # fallback

                action_pre = re.sub(r"<\|box_start\|>|<\|box_end\|>", "", action_pre)
                
                def double_second_value(match):
                    x = int(match.group(1)) * width / 1000
                    y = int(match.group(2)) * height / 1000  # Scale y
                    return f"({int(x)}, {int(y)})"
                
                # Match coordinates with optional comma and whitespace: "123 456" or "123, 456" or "123,456"
                action_pre = re.sub(r"(\d+)(?:,|\s)\s*(\d+)", double_second_value, action_pre)
                action_pre = action_pre.strip()
                self.latest_action = action_pre

                mock_response_dict = parse_action_to_structure_output(action_pre, 1000, height, width, model_type)
                action = convert_coordinates(mock_response_dict, height, width, model_type=model_type)
                self.actions.append(action)

                self.logger.info(f"Thought: {thought}")
                self.logger.info(f"Action: {action}")

                # Execute Action
                stop_flag = execute_action(action, self.adb_path)
                if stop_flag == "STOP":
                    self.running = False
                    break

                # Prepare for next iteration
                last_screenshot_file = os.path.join(self.screenshot_dir, "last_screenshot.png")
                if os.path.exists(last_screenshot_file):
                    os.remove(last_screenshot_file)
                
                # We rename the current screenshot to last_screenshot, but wait...
                # The original code renames it, then calls get_perception_infos again which calls get_screenshot
                # So we lose the 'current' screenshot for a moment.
                # To keep the frontend happy, maybe we should copy it instead?
                # But get_screenshot overwrites 'screenshot.png'.
                # Let's follow original logic for now.
                if os.path.exists(self.screenshot_file):
                     os.rename(self.screenshot_file, last_screenshot_file)

                # Re-acquire screenshot happens at start of loop or here?
                # Original code:
                # rename to last
                # get_perception_infos (takes new screenshot)
                # remove temp
                # mkdir temp
                # remove last
                
                width, height = self.get_perception_infos()
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                os.mkdir(self.temp_dir)
                
                if os.path.exists(last_screenshot_file):
                    os.remove(last_screenshot_file)
                
                # Small sleep to prevent tight loop if errors occur
                time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Error in agent loop: {e}", exc_info=True)
            self.running = False
        finally:
            os.chdir(original_cwd)
            self.logger.info("Agent loop stopped")

    def start(self):
        if not self.running:
            self.running = True # Set running flag immediately
            self.thread = threading.Thread(target=self.run_loop)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
