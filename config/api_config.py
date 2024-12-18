# config/api_config.py
from dataclasses import dataclass
import openai
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    base_url: str
    api_key: str
    model: str = "gpt-4-vision-preview"
    max_retries: int = 3
    retry_delay: int = 2

    def test_connection(self) -> Tuple[bool, str]:
        """测试API连接
        
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        try:
            if not self.validate()[0]:
                return False, "API配置验证失败"
            
            client = openai.OpenAI(
                base_url=self.base_url.strip(),
                api_key=self.api_key.strip()
            )
            
            response = client.chat.completions.create(
                model=self.model.strip(),
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=5
            )
            logger.info("API连接测试成功")
            return True, "API连接测试成功！✅"
            
        except openai.APIError as e:
            logger.error(f"API错误: {str(e)}")
            return False, f"API错误: {str(e)} ❌"
        except openai.RateLimitError:
            logger.error("API请求达到限制")
            return False, "API请求达到限制，请稍后重试 ❌"
        except Exception as e:
            logger.error(f"API连接测试失败: {str(e)}")
            return False, f"API连接测试失败: {str(e)} ❌"

    def validate(self) -> Tuple[bool, str]:
        """验证API配置
        
        Returns:
            Tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            # 验证字段不为空
            self.base_url = self.base_url.strip()
            self.api_key = self.api_key.strip()
            self.model = self.model.strip()
            
            if not all([self.base_url, self.api_key, self.model]):
                return False, "请填写所有API配置信息"
            
            # 验证URL格式
            if not self.base_url.startswith(('http://', 'https://')):
                return False, "API Base URL 格式无效"
            
            # 验证API key格式（可以根据实际需求添加更多验证）
            if len(self.api_key) < 8:
                return False, "API Key 格式无效"
                
            return True, ""
            
        except Exception as e:
            logger.error(f"API配置验证失败: {str(e)}")
            return False, f"API配置验证失败: {str(e)}"
            
    @property
    def headers(self) -> dict:
        """获取API请求头
        
        Returns:
            dict: API请求头
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
