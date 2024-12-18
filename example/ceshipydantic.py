from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import httpx
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

class XAIModel(OpenAIModel):
    """专门的X.AI模型实现"""
    def __init__(
        self,
        model_name: str,
        api_key: str,
        http_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any
    ):
        # 确保基础URL正确
        if http_client is None:
            http_client = httpx.AsyncClient(
                base_url="https://api.x.ai/v1",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0
            )
            
        # 调用父类初始化
        super().__init__(
            model_name,
            api_key=api_key,
            http_client=http_client,
            **kwargs
        )

    def name(self) -> str:
        """获取模型标识符"""
        return f"xai:{self.model_name}"

class TextProcessor:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """初始化文本处理器
        Args:
            base_url: API基础URL
            api_key: API密钥
            model: 模型名称
        """
        # 创建XAI模型实例
        self.model = XAIModel(
            model or "grok-2-1212",
            api_key=api_key
        )
        
        # 1.分析agent：负责分析清洗文本
        self.analyzer_agent = Agent(
            self.model,
            system_prompt="""你是文本分析专家。分析输入的文本块，确保：
            1. 理解其完整语义结构
            2. 识别其主要主题
            3. 保持文本的原始内容

            返回你的理解和分析。""",
            result_type=str
        )

        # 2.标题生成agent：生成简短的instruction
        self.title_agent = Agent(
            self.model,
            system_prompt="""你是标题生成专家。
            为文本生成简短的instruction，要求：
            1. 长度控制在10个字以内
            2. 直接概括文本核心主题
            3. 避免过度解释或分析
            
            例如：
            - "初次绘画"
            - "职业选择"
            - "成人世界"
            
            直接返回标题文本。""",
            result_type=str
        )

        # 3.格式化agent
        self.format_agent = Agent(
            self.model,
            system_prompt="""你是格式化专家。
            请将提供的标题和原文按以下格式组织成JSON：
            {
                "instruction": "标题",   // 简短的概括性指令
                "input": "",    // 保持为空字符串
                "output": "原文"  // 完整的原始文本
            }
            
            注意：
            1. 严格按照格式输出
            2. 保持原文完整
            3. 返回有效的JSON字符串
            """,
            result_type=str
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, 'http_client'):
            await self.model.http_client.aclose()

    async def split_paragraphs(self, text: str) -> List[str]:
        """分割文本为段落。"""
        paragraphs = []
        current = []

        lines = [line.strip() for line in text.split('\n')]
        
        for line in lines:
            is_chapter = any(char.isdigit() for char in line[:2]) if line else False
            
            if (not line or is_chapter) and current:
                paragraphs.append(' '.join(current))
                current = []
            
            if is_chapter:
                if current:
                    paragraphs.append(' '.join(current))
                    current = []
                current.append(line)
            elif line:
                current.append(line)
                
        if current:
            paragraphs.append(' '.join(current))
            
        return [p for p in paragraphs if len(p.strip()) > 50 or any(char.isdigit() for char in p[:2])]

    async def process_paragraph(self, paragraph: str) -> Dict:
        """处理单个段落"""
        try:
            analysis_result = await self.analyzer_agent.run(paragraph)
            title_result = await self.title_agent.run(paragraph)
            prompt = f"""
            标题：{title_result.data}
            原文：{paragraph}
            请按指定格式生成JSON。
            """
            formatted_result = await self.format_agent.run(prompt)
            return json.loads(formatted_result.data)
        except Exception as e:
            print(f"处理段落时出错: {str(e)}")
            return {
                "instruction": "待处理文本",
                "input": "",
                "output": paragraph
            }

    async def process_file(self, input_file: Path, output_file: Path) -> None:
        """处理输入文件"""
        # 读取文件
        text = input_file.read_text(encoding='utf-8')
        
        # 分割段落
        paragraphs = await self.split_paragraphs(text)
        
        dataset = []
        for i, paragraph in enumerate(paragraphs, 1):
            print(f"处理段落 {i}/{len(paragraphs)}")
            result = await self.process_paragraph(paragraph)
            if result:
                dataset.append(result)
            
            # 每处理3个段落保存一次
            if i % 3 == 0:
                self.save_interim_results(dataset, output_file)

        # 保存最终结果
        self.save_interim_results(dataset, output_file)
        print(f"处理完成，共生成 {len(dataset)} 条数据")

    @staticmethod
    def save_interim_results(dataset: List[dict], output_file: Path) -> None:
        """保存中间结果"""
        output_file.write_text(
            json.dumps(dataset, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

# 使用示例
async def main():
    # API配置
    config = {
        "api_key": "xai-",  # 替换成你的API密钥
        "model": "grok-2-1212"
    }
    
    input_file = Path("民法典前一章.txt")  
    output_file = Path("11dataset.json")
    output_file.parent.mkdir(exist_ok=True)
    
    async with TextProcessor(**config) as processor:
        try:
            await processor.process_file(input_file, output_file)
        except KeyboardInterrupt:
            print("\n处理被中断")
        except Exception as e:
            print(f"处理出错: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())