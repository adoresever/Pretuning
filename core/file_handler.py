# core/file_handler.py
import os
import shutil
from typing import List
from PIL import Image

class FileSystemHandler:
    def __init__(self, temp_dir: str = "temp_dataset"):
        self.temp_dir = os.path.abspath(temp_dir)
        self.ensure_temp_dir()

    def ensure_temp_dir(self) -> None:
        os.makedirs(self.temp_dir, exist_ok=True)

    def get_temp_path(self, filename: str) -> str:
        return os.path.join(self.temp_dir, filename)

    def save_temp_image(self, img: Image.Image, save_path: str) -> None:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)

    def cleanup(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        self.ensure_temp_dir()

    @staticmethod
    def get_image_files(directory: str) -> List[str]:
        # 确保使用绝对路径
        directory = os.path.abspath(directory)
        image_extensions = {'.png', '.jpg', '.jpeg'}
        
        # 检查目录是否存在
        if not os.path.exists(directory):
            print(f"目录不存在: {directory}")
            return []
            
        return [f for f in os.listdir(directory) 
                if os.path.splitext(f)[1].lower() in image_extensions]