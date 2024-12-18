
from pathlib import Path
import openai
import time
import httpx
from typing import List, Dict
from config.api_config import APIConfig
from core.image_processor import ImageProcessor
from pydantic_ai.models.openai import OpenAIModel

class APIHandler:
    def __init__(self, config: APIConfig):
        """初始化API处理器
        
        Args:
            config: API配置对象
        """
        self.config = config
        
        self.client = openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key
        )
        
        
        self.http_client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=30.0
        )
        
        self.pydantic_model = OpenAIModel(
            config.model,
            api_key=config.api_key,
            http_client=self.http_client
        )
        
        self.max_retries = 3
        self.retry_delay = 2
        self.system_prompt = "你是一个专业的图像识别专家。请详细描述这张医学图像。"
    
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
        
    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词
        
        Args:
            prompt: 提示词内容
        """
        self.system_prompt = prompt

    def generate_description(self, image_path: Path, retry_count: int = 0) -> str:
        """生成图片描述
        
        Args:
            image_path: 图片路径
            retry_count: 重试次数
            
        Returns:
            str: 生成的描述文本
        """
        try:
            # 准备消息
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]
            
            # 编码图片
            base64_image = ImageProcessor.encode_image(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            })

            # 调用API
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages
            )
            
            return response.choices[0].message.content

        except openai.RateLimitError as e:
            # 达到速率限制时重试
            if retry_count < self.max_retries:
                time.sleep(self.retry_delay * (2 ** retry_count))
                return self.generate_description(image_path, retry_count + 1)
            return f"生成失败: {str(e)}"
            
        except Exception as e:
            error_msg = str(e).encode('utf-8').decode('utf-8')
            return f"生成失败: {error_msg}"