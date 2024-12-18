# core/create_parquet.py
from pathlib import Path
import os
import json
from PIL import Image
from datasets import Dataset
from tqdm import tqdm
from typing import Optional, Dict, List
import time

from core.image_processor import ImageProcessor
from core.file_handler import FileSystemHandler

class DatasetProcessor:
    def __init__(self, fs_handler: Optional[FileSystemHandler] = None):
        self.fs_handler = fs_handler or FileSystemHandler()

    def create_from_pairs(self, image_text_pairs: List[Dict], output_dir: str = "image_dataset") -> Dataset:  # 修改这里的默认值
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(output_dir) / f"dataset_{timestamp}"
            output_path.mkdir(parents=True, exist_ok=True)
            
            metadata = []
            for i, pair in enumerate(image_text_pairs):
                img_filename = f"image_{i:03d}.png"
                img_path = output_path / img_filename
                pair['image'].save(img_path)
                
                metadata.append({
                    'image_file': img_filename,
                    'text': pair['text']
                })
            
            metadata_file = output_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            dataset = Dataset.from_dict({
                'image': [pair['image'] for pair in image_text_pairs],
                'text': [pair['text'] for pair in image_text_pairs]
            })
            
            dataset.save_to_disk(output_path / "hf_dataset")
            return dataset
            
        except Exception as e:
            raise

    @staticmethod
    def load_dataset(dataset_path: str) -> Dict:
        try:
            path = Path(dataset_path)
            with open(path / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            dataset = []
            for item in metadata:
                img_path = path / item['image_file']
                img = Image.open(img_path)
                dataset.append({
                    'image': img,
                    'text': item['text']
                })
                
            return {
                'samples': dataset,
                'metadata': metadata,
                'path': str(path)
            }
            
        except Exception as e:
            raise

    @staticmethod
    def verify_dataset(dataset_path: str) -> None:
        try:
            dataset = DatasetProcessor.load_dataset(dataset_path)
            if dataset['samples']:
                sample = dataset['samples'][0]
                print(f"数据集路径: {dataset['path']}")
                print(f"样本数量: {len(dataset['samples'])}")
                print(f"图片尺寸: {sample['image'].size}")
                
        except Exception as e:
            raise