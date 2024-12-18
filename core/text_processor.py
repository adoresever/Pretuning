from pathlib import Path
from typing import List, Dict, Tuple
import json
import time
from tqdm import tqdm
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
import httpx
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, api_handler=None):
        self.api_handler = api_handler
        self.text_results = []
        self.http_client = None
        self.model = None
        self._initialize_agents()

    async def __aenter__(self):
        if not self.http_client or self.http_client.is_closed:
            self._initialize_agents()  
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None  

    def _initialize_agents(self):
        try:
            if not self.api_handler:
                self.analyzer_agent = None
                self.title_agent = None
                self.format_agent = None
                return

            # 创建 HTTP 客户端
            self.http_client = httpx.AsyncClient(
                base_url=self.api_handler.config.base_url,
                headers={"Authorization": f"Bearer {self.api_handler.config.api_key}"},
                timeout=30.0
            )
            
            # 创建 OpenAI 模型实例
            self.model = OpenAIModel(
                self.api_handler.config.model,
                api_key=self.api_handler.config.api_key,
                http_client=self.http_client
            )

            # 创建 agents
            self.analyzer_agent = Agent(
                self.model,
                system_prompt="""你是文本分析专家。
                你的任务是分析输入的文本，在接近1000个token的位置找到合适的语义断点，这个断点应该尽量保持段落或语义的完整性。
                
                要求：
                1. 寻找最接近1000 token处的语义完整位置
                2. 优先在段落结束处断开
                3. 返回从开始到断点的完整文本
                4. 关注语义连贯性，不要在句子中间断开
                
                直接返回这段完整的文本。""",
                result_type=str
            )

            self.title_agent = Agent(
                self.model,
                system_prompt="""你是标题生成专家。
                为文本生成简短的instruction，要求：
                1. 长度控制在10个字以内
                2. 直接概括文本核心主题
                3. 避免过度解释或分析

                直接返回标题文本。""",
                result_type=str
            )

            self.format_agent = Agent(
            self.model,
            system_prompt="""你是格式化专家。
                请将提供的标题和原文按以下格式组织成JSON（直接返回 JSON，不要包含任何其他标记）：
                {
                    "instruction": "标题",
                    "input": "",
                    "output": "原文"
                }
                
                注意：
                1. 严格按照格式输出
                2. 保持原文完整
                3. 只返回纯 JSON 字符串，不要包含 markdown 代码块标记
                4. 确保 JSON 中的换行使用 \n
                """,
                result_type=str
            )
            
            logger.info("Agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise

    async def update_prompts(self, analyzer_prompt: str, title_prompt: str, format_prompt: str):
        try:
            print("开始更新提示词...")
            if not self.api_handler:
                raise Exception("请先配置API设置")
            
            if not self.model:
                raise Exception("Model未初始化")

            # 关闭现有连接
            if self.http_client:
                await self.http_client.aclose()
                
            # 重新创建 HTTP 客户端
            self.http_client = httpx.AsyncClient(
                base_url=self.api_handler.config.base_url,
                headers={"Authorization": f"Bearer {self.api_handler.config.api_key}"},
                timeout=30.0
            )
            
            # 重新创建模型实例
            self.model = OpenAIModel(
                self.api_handler.config.model,
                api_key=self.api_handler.config.api_key,
                http_client=self.http_client
            )

            print("更新analyzer agent...")
            self.analyzer_agent = Agent(
                self.model,
                system_prompt=analyzer_prompt.strip(),
                result_type=str
            )
            
            print("更新title agent...")
            self.title_agent = Agent(
                self.model,
                system_prompt=title_prompt.strip(),
                result_type=str
            )
            
            print("更新format agent...")
            self.format_agent = Agent(
                self.model,
                system_prompt=format_prompt.strip(),
                result_type=str
            )
            
            print("所有提示词更新完成")
            
        except Exception as e:
            import traceback
            print(f"更新提示词失败: {str(e)}")
            print(traceback.format_exc())
            raise

    async def process_paragraph(self, paragraph: str) -> Dict:
        try:
            if not all([self.analyzer_agent, self.title_agent, self.format_agent]):
                raise Exception("请先配置API设置")

            print("开始分析文本...")
            analysis = await self.analyzer_agent.run(paragraph)
            print(f"分析结果: {analysis.data[:100]}...")

            print("生成标题...")
            title = await self.title_agent.run(paragraph)
            print(f"生成的标题: {title.data}")

            print("格式化结果...")
            prompt = f"""
            标题：{title.data}
            原文：{paragraph}
            请按指定格式生成JSON。
            """
            formatted = await self.format_agent.run(prompt)
            print(f"格式化完成: {formatted.data[:100]}...")

            try:
                # 移除可能存在的 markdown 标记
                json_str = formatted.data.replace('```json', '').replace('```', '').strip()
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"JSON 解析错误: {e}")
                return {
                    "instruction": title.data if title and hasattr(title, 'data') else "待处理文本",
                    "input": "",
                    "output": paragraph
                }
                
        except Exception as e:
            print(f"处理段落出错: {str(e)}")
            return {
                "instruction": "待处理文本",
                "input": "",
                "output": paragraph
            }

    async def process_file(self, content: str) -> Tuple[str, str]:
        try:
            if not self.model:
                print("错误: model未初始化")
                return "", "请先配置API设置"

            print(f"开始处理文件，内容长度: {len(content)}")
            print("文件内容前100字符:")
            print(content[:100])

            # 使用分块 agent 处理文本
            current_content = content
            paragraphs = []
            
            while len(current_content) > 100:  # 设置最小长度阈值
                result = await self.analyzer_agent.run(current_content)
                split_text = result.data
                
                if not split_text or len(split_text) < 50:  # 防止过短分割
                    break
                    
                paragraphs.append(split_text)
                # 更新剩余内容
                current_content = current_content[len(split_text):].strip()
            
            # 处理剩余内容
            if current_content:
                paragraphs.append(current_content)

            if not paragraphs:
                return "", "未找到有效文本块"

            self.text_results = []
            preview = ""
            
            for i, paragraph in enumerate(paragraphs, 1):
                result = await self.process_paragraph(paragraph)
                self.text_results.append(result)
                preview += f"=== 文本块 {i} ===\n"
                preview += f"指令: {result.get('instruction', '待处理')}\n"
                preview += f"输出: {result.get('output', paragraph)[:100]}...\n\n"
            
            return preview, "处理完成"
                
        except Exception as e:
            print(f"处理文件出错: {str(e)}")
            return "", f"处理失败: {str(e)}"

            
    def save_dataset(self, output_path: str) -> str:
        try:
            if not self.text_results:
                print("没有可保存的数据")
                return "没有可保存的数据"
            
            # 使用 os.path 处理路径
            import os
            output_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存文件
            print(f"保存数据集到: {output_path}")
            print(f"数据条数: {len(self.text_results)}")
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.text_results, f, ensure_ascii=False, indent=2)
                print(f"成功保存 {len(self.text_results)} 条数据")
                return f"已保存 {len(self.text_results)} 条数据到 {output_path}"
                
            except Exception as e:
                print(f"写入文件失败: {str(e)}")
                return f"保存失败: {str(e)}"
                
        except Exception as e:
            import traceback
            print(f"保存数据集失败: {str(e)}")
            print(traceback.format_exc())
            return f"保存失败: {str(e)}"