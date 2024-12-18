# core/dataset_creator.py
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
from datasets import Dataset
import time
import json

from core.text_processor import TextProcessor
from core.image_processor import ImageProcessor
from core.file_handler import FileSystemHandler
from core.api_handler import APIHandler
from config.api_config import APIConfig

class DatasetCreator:
    def __init__(self):
        self.fs_handler = FileSystemHandler()
        self.image_text_pairs = []
        self.api_handler: Optional[APIHandler] = None
        self.example_count = 2
        self.text_processor = None
        self.text_results = []

    def save_text_dataset(self, output_dir: str = "text_dataset") -> str:
        try:
            if not self.text_processor or not self.text_processor.text_results:
                print("没有可保存的数据")
                return "没有可保存的数据"
                
            # 确保输出目录存在
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"dataset_{timestamp}.json"
            output_path = os.path.join(output_dir, output_file)
            
            print(f"保存数据集到: {output_path}")
            print(f"数据条数: {len(self.text_processor.text_results)}")
            
            # 保存文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.text_processor.text_results, f, ensure_ascii=False, indent=2)
            print("文件保存成功")
            
            return f"已保存 {len(self.text_processor.text_results)} 条数据到 {output_path}"
                
        except Exception as e:
            print(f"保存数据集过程出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"保存失败: {str(e)}"

    def set_api_config(self, base_url: str, api_key: str, model: str) -> str:
        try:
            print("开始配置API...")  # 添加调试信息
            config = APIConfig(
                base_url=base_url.strip(),
                api_key=api_key.strip(),
                model=model.strip()
            )
            
            is_valid, error_msg = config.validate()
            if not is_valid:
                print(f"API配置验证失败: {error_msg}")  # 添加调试信息
                return error_msg
                    
            print("创建API处理器...")  # 添加调试信息
            # 创建新的API处理器
            self.api_handler = APIHandler(config)
            
            print("更新文本处理器...")  # 添加调试信息
            # 更新或创建文本处理器
            if not self.text_processor:
                self.text_processor = TextProcessor(self.api_handler)
                print("创建了新的文本处理器")  # 添加调试信息
            else:
                # 更新现有文本处理器的API处理器
                self.text_processor.api_handler = self.api_handler
                self.text_processor._initialize_agents()
                print("更新了现有文本处理器")  # 添加调试信息
                    
            return "API配置已更新 ✅"
        except Exception as e:
            import traceback
            print(f"API配置失败: {str(e)}")
            print(traceback.format_exc())
            return f"API配置失败: {str(e)} ❌"

    def test_api_connection(self) -> str:
        if not self.api_handler:
            return "请先配置API设置"
        success, message = self.api_handler.config.test_connection()
        return message

    def process_images(self, files) -> Tuple[List[Image.Image], List[List]]:
        images = []
        text_data = []
        
        try:
            for file in files:
                try:
                    # 使用 os.path 处理文件路径
                    import os
                    file_path = os.path.abspath(file.name)
                    img = ImageProcessor.load_and_preprocess(file_path)
                    img = ImageProcessor.resize_image(img)
                    
                    index = len(self.image_text_pairs)
                    save_path = self.fs_handler.get_temp_path(f"image_{index}.png")
                    self.fs_handler.save_temp_image(img, save_path)
                    
                    self.image_text_pairs.append({
                        'index': int(index),
                        'image': img,
                        'image_path': save_path,
                        'text': ""
                    })
                    
                    images.append(img)
                    text_data.append([int(index), ""])
                    
                except Exception as e:
                    print(f"处理图片失败: {str(e)}")
                    continue
            
            return images, text_data
            
        except Exception as e:
            print(f"批量处理图片失败: {str(e)}")
            return [], []

    def batch_generate_all(self, prompt_template: str = None) -> Tuple[List[List], str]:
        """批量为所有图片生成描述
        
        Args:
            prompt_template: 用于生成描述的提示词模板
            
        Returns:
            Tuple[List[List], str]: 包含 [[index, description], ...] 的列表和状态消息
        """
        try:
            if not self.api_handler:
                print("错误: API处理器未初始化")
                return [], "请先配置API设置"
                
            if not self.image_text_pairs:
                print("错误: 未找到图片数据")
                return [], "请先上传图片"

            text_data = []
            print(f"开始处理 {len(self.image_text_pairs)} 张图片...")
            
            for pair in self.image_text_pairs:
                try:
                    index = pair['index']
                    print(f"处理图片 {index}: {pair['image_path']}")
                    
                    # 如果提供了新的提示词，更新系统提示词
                    if prompt_template:
                        self.api_handler.set_system_prompt(prompt_template)
                    
                    # 生成描述
                    description = self.api_handler.generate_description(pair['image_path'])
                    pair['text'] = description
                    text_data.append([index, description])
                    print(f"完成图片 {index} 的描述生成")
                    
                except Exception as e:
                    print(f"处理图片 {pair['index']} 失败: {str(e)}")
                    text_data.append([pair['index'], ""])
            
            # 确保返回的数据列表长度为10
            while len(text_data) < 10:
                text_data.append([len(text_data), ""])
                
            success_count = sum(1 for item in text_data if item[1].strip())
            message = f"已完成 {success_count}/{len(text_data)} 张图片的描述生成"
            print(message)
            
            return text_data, message
            
        except Exception as e:
            print(f"批量生成描述失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return [], f"生成失败: {str(e)}"
    def create_dataset(self) -> str:
        try:
            if not self.image_text_pairs:
                return "没有数据可保存"
            
            empty_texts = [str(p['index']) for p in self.image_text_pairs if not p['text'].strip()]
            if empty_texts:
                return f"以下编号的图片缺少描述: {', '.join(empty_texts)}"
            
            dataset_dict = {
                'image': [pair['image'] for pair in self.image_text_pairs],
                'text': [pair['text'] for pair in self.image_text_pairs]
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            # 使用 os.path 处理路径
            import os
            output_path = os.path.abspath("image_dataset")
            os.makedirs(output_path, exist_ok=True)
            dataset.save_to_disk(output_path)
            
            self.verify_dataset()
            
            return "数据集保存成功 ✅"
            
        except Exception as e:
            return f"保存数据集失败: {str(e)} ❌"

    def verify_dataset(self) -> str:
        try:
            dataset = Dataset.load_from_disk("image_dataset")
            print(f"数据集样本数: {len(dataset)}")
            return "数据集验证完成 ✅"
            
        except Exception as e:
            return f"验证失败: {str(e)} ❌"
    
    def update_text(self, text_data: List[List]) -> str:
        try:
            for index, text in text_data:
                for pair in self.image_text_pairs:
                    if pair['index'] == index:
                        pair['text'] = text
            return "文本更新成功"
        except Exception as e:
            return f"更新失败: {str(e)}"
            
    def save_text_dataset(self, output_dir: str = "text_dataset") -> str:
        try:
            if not self.text_processor or not self.text_processor.text_results:
                return "没有可保存的数据"
                
            # 使用 os.path 处理路径
            import os
            
            # 确保输出目录存在
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"dataset_{timestamp}.json"
            output_path = os.path.join(output_dir, output_file)
            
            print(f"保存数据集到: {output_path}")
            print(f"数据条数: {len(self.text_processor.text_results)}")
            
            # 保存文件
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(self.text_processor.text_results, f, ensure_ascii=False, indent=2)
                print("文件保存成功")
                return f"已保存 {len(self.text_processor.text_results)} 条数据到 {output_path}"
            except Exception as e:
                print(f"保存文件失败: {str(e)}")
                return f"保存失败: {str(e)}"
                
        except Exception as e:
            print(f"保存数据集过程出错: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"保存失败: {str(e)}"

    def test_single_image_description(self, index: int = 0) -> str:
        try:
            if not self.api_handler or not self.image_text_pairs:
                return "请先配置API并上传图片"
            
            if index >= len(self.image_text_pairs):
                return "图片索引超出范围"
                
            pair = self.image_text_pairs[index]
            description = self.api_handler.generate_description(pair['image_path'])
            pair['text'] = description
            
            return f"测试成功: {description[:100]}..."
            
        except Exception as e:
            return f"测试失败: {str(e)}"