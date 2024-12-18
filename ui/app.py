import gradio as gr
from core.dataset_creator import DatasetCreator

def create_ui():
    creator = DatasetCreator()
    
    with gr.Blocks() as app:
        gr.Markdown("# Pretuning 微调数据集创建工具")
        
        with gr.Tab("API设置"):
            with gr.Row():
                with gr.Column(scale=2):
                    api_base = gr.Textbox(
                        label="API Base URL",
                        placeholder="输入API基础URL",
                        value="https://api.openai.com/v1",  
                        type="text"
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        placeholder="输入您的API密钥",
                        type="password"
                    )
                    model = gr.Textbox(
                        label="模型名称",
                        value="gpt-4o-mini",  # 默认值
                        placeholder="例如: gpt-4-vision-preview"
                    )
        

            with gr.Row():
                save_api = gr.Button("保存设置", variant="primary")
                test_api = gr.Button("测试连接", variant="secondary")
            api_status = gr.Textbox(label="API状态", interactive=False)
        
        with gr.Tab("文本数据处理"):
            gr.Markdown("""
            ### 使用说明
            1. 配置API设置（PydanticAI受限优先使用openai）
            2. 设置三个处理Agent的提示词
            3. 上传文本文件进行处理
            4. 检查生成的结果并保存数据集
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Agent设置")
                    analyzer_prompt = gr.Textbox(
                        label="分析器 Agent 提示词",
                        placeholder="设置文本分析专家的提示词",
                        value="""你是文本分析专家。
                        你的任务是分析输入的文本，在接近1000个token的位置找到合适的语义断点，这个断点应该尽量保持段落或语义的完整性。
                        
                        要求：
                        1. 寻找最接近1000 token处的语义完整位置
                        2. 优先在段落结束处断开
                        3. 返回从开始到断点的完整文本
                        4. 关注语义连贯性，不要在句子中间断开
                        
                        直接返回这段完整的文本。""",
                        lines=10
                    )
                        
                    title_prompt = gr.Textbox(
                        label="标题生成器 Agent 提示词",
                        placeholder="设置标题生成专家的提示词",
                        value="""你是标题生成专家。
                        为文本生成简短的instruction，要求：
                        1. 长度控制在10个字以内
                        2. 直接概括文本核心主题
                        3. 避免过度解释或分析
                        
                        例如：
                        - "初次绘画"
                        - "职业选择"
                        - "成人世界"
                        
                        直接返回标题文本。""",
                        lines=8
                    )
                    
                    format_prompt = gr.Textbox(
                        label="格式化器 Agent 提示词",
                        placeholder="设置格式化专家的提示词",
                        value="""你是格式化专家。
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
                        lines=10
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 处理结果")
                    output_text = gr.TextArea(
                        label="处理结果预览",
                        interactive=False,
                        lines=35
                    )
            
            with gr.Row():
                process_text = gr.Button("处理文本", variant="secondary")
                save_text = gr.Button("保存数据集", variant="primary")
            
            status = gr.Textbox(label="状态", interactive=False)
            
            gr.Markdown("---")
            with gr.Row():
                text_file = gr.File(
                    label="上传文本文件",
                    file_types=[".txt"],
                    file_count="single"
                )

            # 文本处理相关函数
            async def handle_text_processing(text_file, analyzer_prompt, title_prompt, format_prompt):
                try:
                    # 1. 基本检查
                    if not text_file:
                        print("错误: 未上传文件")
                        return "", "请先上传文件"
                    
                    if not creator.text_processor:
                        print("错误: 未配置text_processor")
                        return "", "请先配置API设置"
                    
                    # 2. 读取文件
                    try:
                        if isinstance(text_file, str):
                            print(f"读取文件路径: {text_file}")
                            with open(text_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                        else:
                            print(f"读取上传文件: {text_file.name}")
                            with open(text_file.name, 'r', encoding='utf-8') as f:
                                content = f.read()
                        print(f"成功读取文件，内容长度: {len(content)} 字符")
                        
                    except Exception as e:
                        print(f"读取文件失败: {e}")
                        return "", f"读取文件失败: {str(e)}"
                    
                    # 3. 在每次处理前都重新初始化连接和更新提示词
                    print("\n=== 重新初始化处理器 ===")
                    try:
                        # 确保关闭之前的连接
                        if creator.text_processor.http_client:
                            await creator.text_processor.http_client.aclose()
                        
                        # 重新初始化
                        creator.text_processor._initialize_agents()
                        
                        print("更新提示词...")
                        await creator.text_processor.update_prompts(
                            analyzer_prompt=analyzer_prompt,
                            title_prompt=title_prompt,
                            format_prompt=format_prompt
                        )
                        print("处理器更新成功")
                    except Exception as e:
                        print(f"处理器更新失败: {e}")
                        return "", f"处理器更新失败: {str(e)}"
                    
                    # 4. 使用异步上下文管理器处理文本
                    print("\n=== 处理文本 ===")
                    try:
                        async with creator.text_processor as processor:
                            preview, message = await processor.process_file(content)
                            print(f"处理完成: {message}")
                            return preview, message
                            
                    except Exception as e:
                        print(f"文本处理失败: {e}")
                        import traceback
                        print(traceback.format_exc())
                        # 重新初始化处理器
                        creator.text_processor._initialize_agents()
                        return "", f"文本处理失败: {str(e)}"
                        
                except Exception as e:
                    print(f"处理过程出现异常: {e}")
                    import traceback
                    print(traceback.format_exc())
                    return "", f"处理失败: {str(e)}"

            async def handle_save_text_dataset():
                print("开始保存文本数据集...")
                try:
                    result = creator.save_text_dataset()
                    print(f"保存结果: {result}")
                    return result
                except Exception as e:
                    print(f"保存数据集失败: {str(e)}")
                    return f"保存失败: {str(e)}"
            # 修改事件绑定部分，确保使用异步处理
            process_text.click(
                fn=handle_text_processing,
                inputs=[text_file, analyzer_prompt, title_prompt, format_prompt],
                outputs=[output_text, status],
                api_name="process_text"
            )

            save_text.click(
                fn=handle_save_text_dataset,
                outputs=[status]
            )

        with gr.Tab("图像数据处理"):
            gr.Markdown("""
            ### 使用说明
            1. 上传需要处理的图片
            2. 设置描述的提示词和风格
            3. 点击生成描述，将自动为所有图片生成描述
            4. 必须选择多模态视觉识别的模型！
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    preview_image = gr.Image(
                        label="图片预览",
                        type="pil",
                        height=400
                    )
                    
                    # 添加提示词输入
                    prompt_template = gr.Textbox(
                        label="描述提示词",
                        placeholder="例如：请详细描述这张旅游照片，包含以下方面：1. 图像类型 2. 主要特征 3. 关键医学指标",
                        lines=3,
                        value="你是一位三亚地区景点分析专家，请分析图片：\n1. 图片中的自然特征\n2. 建筑风格\n3. 人文元素\n4. 地域植被特征。\n图片的地域来自三亚，围绕这个进行分析"
                    )
                
                with gr.Column(scale=1):
                    text_boxes = gr.Dataframe(
                        headers=["编号", "描述文本"],
                        datatype=["number", "str"],
                        interactive=True,
                        row_count=10,
                        wrap=True,
                        col_count=(2, "fixed")
                    )
            
            with gr.Row():
                batch_generate = gr.Button("批量生成描述", variant="secondary")
                test_llm = gr.Button("测试LLM描述", variant="secondary")
                save_button = gr.Button("保存数据集", variant="primary")
                verify_button = gr.Button("验证数据集", variant="secondary")  

            
            status = gr.Textbox(label="状态", interactive=False)
            
            gr.Markdown("---")
            with gr.Row():
                file_output = gr.Files(
                    label="上传图片",
                    file_count="multiple",
                    elem_classes="upload-box"
                )

            def handle_upload(files):
                if not files:
                    return None, None
                try:
                    print("开始处理上传的文件...")
                    images, text_data = creator.process_images(files)
                    print(f"处理的图片数量: {len(images)}")
                    print(f"处理的文本数据: {text_data}")
                    if not images:
                        return None, None
                    return images[0], text_data
                except Exception as e:
                    print(f"上传处理错误: {str(e)}")
                    return None, None

            def handle_text_update(data):

                try:
                    
                    
                    # 确保数据是字典格式
                    if isinstance(data, dict) and 'data' in data:
                        rows = data['data']
                        valid_data = []
                        
                        # 处理每一行数据
                        for index, text in rows:
                            if str(index).strip() and str(index).isdigit():
                                # 转换为正确的格式
                                valid_data.append([int(index), str(text) if text else ""])
                                
                        
                        if valid_data:
                            return creator.update_text(valid_data)
                    
                    return "无效的数据格式"
                    
                except Exception as e:
                  
                    return f"更新失败: {str(e)}"

            def handle_preview_update(evt: gr.SelectData):
                try:
                    row_index = evt.index[0]  # 获取选中行的索引
                    if 0 <= row_index < len(creator.image_text_pairs):
                        return creator.image_text_pairs[row_index]['image']
                    return None
                except Exception as e:
                    print(f"预览更新错误: {str(e)}")
                    return None
            
            def handle_batch_generate(prompt):
                try:
                    print(f"[handle_batch_generate] Using prompt: {prompt}")
                    
                    if not creator.api_handler:
                        return [], "请先配置API设置"
                        
                    # 清理之前的临时文件
                    creator.fs_handler.ensure_temp_dir()
                    
                    # 更新系统提示词
                    if prompt and creator.api_handler:
                        creator.api_handler.set_system_prompt(prompt)
                        
                    # 生成描述
                    text_data, message = creator.batch_generate_all(prompt)
                    return text_data, message
                except Exception as e:
                    print(f"[handle_batch_generate] Error: {str(e)}")
                    return [], str(e)

            def handle_save_dataset():
                try:
                    return creator.create_dataset()
                except Exception as e:
                    print(f"保存数据集错误: {str(e)}")
                    return str(e)

            def handle_test_llm():
                try:
                    return creator.test_single_image_description(0)
                except Exception as e:
                    print(f"[handle_test_llm] Error: {str(e)}")
                    return f"测试失败: {str(e)}"
            def handle_verify_dataset():
                try:
                    return creator.verify_dataset()
                except Exception as e:
                    print(f"验证处理错误: {str(e)}")
                    return str(e)

            # 绑定事件
            file_output.upload(
                fn=handle_upload,
                inputs=[file_output],
                outputs=[preview_image, text_boxes]
            )
            
            text_boxes.change(
                fn=handle_text_update,
                inputs=[text_boxes],
                outputs=[status]
            )
                        
            text_boxes.select(
                fn=handle_preview_update,
                outputs=[preview_image]
            )
            
            batch_generate.click(
                fn=handle_batch_generate,
                inputs=[prompt_template],
                outputs=[text_boxes, status]
            )
                        
            save_button.click(
                fn=handle_save_dataset,
                outputs=[status]
            )
            
            save_api.click(
                fn=creator.set_api_config,
                inputs=[api_base, api_key, model],
                outputs=[api_status]
            )

            test_api.click(
                fn=creator.test_api_connection,
                outputs=[api_status]
            )
            
            test_llm.click(
                fn=handle_test_llm,
                outputs=[status]
            )
            verify_button.click(
                fn=handle_verify_dataset,
                outputs=[status]
            )

        # 样式设置
        app.style = """
            .upload-box {
                max-height: 120px !important;
                overflow-y: auto !important;
                margin-top: 10px !important;
                padding: 10px !important;
                border: 1px dashed #ccc !important;
                border-radius: 4px !important;
            }
        """
            
    return app
