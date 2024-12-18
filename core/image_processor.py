# core/image_processor.py
import base64
from pathlib import Path
from PIL import Image
from typing import Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    DEFAULT_MAX_SIZE = (800, 600)
    DATASET_SIZE = (160, 40)

    @staticmethod
    def resize_image(img: Image.Image, max_size: Tuple[int, int] = DEFAULT_MAX_SIZE) -> Image.Image:
        """保持比例调整图片大小
        
        Args:
            img: 输入图片
            max_size: 最大尺寸 (宽, 高)
            
        Returns:
            调整大小后的图片
        """
        try:
            width, height = img.size
            if width <= max_size[0] and height <= max_size[1]:
                return img
                
            ratio = min(max_size[0]/width, max_size[1]/height)
            new_size = (int(width * ratio), int(height * ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"调整图片大小时出错: {str(e)}")
            raise

    @staticmethod
    def resize_for_dataset(img: Image.Image) -> Image.Image:
        """调整图片大小为数据集所需的尺寸
        
        Args:
            img: 输入图片
            
        Returns:
            调整后的RGB图片
        """
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img.resize(ImageProcessor.DATASET_SIZE, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"调整数据集图片大小时出错: {str(e)}")
            raise

    @staticmethod
    def encode_image(image_path: Union[str, Path]) -> str:
        """将图片转换为base64编码
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64编码的图片字符串
        """
        try:
            image_path = Path(image_path)
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败: {str(e)}")
            raise

    @staticmethod
    def load_and_preprocess(image_path: Union[str, Path]) -> Image.Image:
        """加载并预处理图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            预处理后的RGB图片
        """
        try:
            image_path = Path(image_path)
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
        except Exception as e:
            logger.error(f"图片加载预处理失败: {str(e)}")
            raise

    @staticmethod
    def validate_image(image_path: Union[str, Path]) -> bool:
        """验证图片是否有效
        
        Args:
            image_path: 图片路径
            
        Returns:
            bool: 图片是否有效
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return False
            Image.open(image_path)
            return True
        except Exception:
            return False