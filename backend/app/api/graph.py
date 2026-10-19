"""
图谱相关API路由
采用项目上下文机制，服务端持久化状态
"""

import os
import traceback
import threading
from flask import request, jsonify

from . import graph_bp
from ..config import Config
from ..services.ontology_generator import OntologyGenerator
from ..services.graphrag_builder import GraphBuilderService  # ← GraphRAG 版本
from ..services.text_processor import TextProcessor
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from ..models.task import TaskManager, TaskStatus
from ..models.project import ProjectManager, ProjectStatus

# 获取日志器
logger = get_logger('weaveragent.api')


def allowed_file(filename: str) -> bool:
    """检查文件扩展名是否允许"""
    if not filename or '.' not in filename:
        return False
    ext = os.path.splitext(filename)[1].lower().lstrip('.')
    return ext in Config.ALLOWED_EXTENSIONS


# ============== 项目管理接口 ==============

@graph_bp.route('/project/<project_id>', methods=['GET'])
def get_project(project_id: str):
    """
    获取项目详情
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"项目不存在: {project_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "data": project.to_dict()
    })


@graph_bp.route('/project/list', methods=['GET'])
def list_projects():
    """
    列出所有项目
    """
    limit = request.args.get('limit', 50, type=int)
    projects = ProjectManager.list_projects(limit=limit)
    
    return jsonify({
        "success": True,
        "data": [p.to_dict() for p in projects],
        "count": len(projects)
    })


@graph_bp.route('/project/<project_id>', methods=['DELETE'])
def delete_project(project_id: str):
    """
    删除项目
    """
    success = ProjectManager.delete_project(project_id)
    
    if not success:
        return jsonify({
            "success": False,
            "error": f"项目不存在或删除失败: {project_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "message": f"项目已删除: {project_id}"
    })


@graph_bp.route('/project/<project_id>/reset', methods=['POST'])
def reset_project(project_id: str):
    """
    重置项目状态（用于重新构建图谱）
    """
    project = ProjectManager.get_project(project_id)
    
    if not project:
        return jsonify({
            "success": False,
            "error": f"项目不存在: {project_id}"
        }), 404
    
    # 重置到本体已生成状态
    if project.ontology:
        project.status = ProjectStatus.ONTOLOGY_GENERATED
    else:
        project.status = ProjectStatus.CREATED
    
    project.graph_id = None
    project.graph_build_task_id = None
    project.error = None
    ProjectManager.save_project(project)
    
    return jsonify({
        "success": True,
        "message": f"项目已重置: {project_id}",
        "data": project.to_dict()
    })


# ============== 接口1：上传文件并生成本体 ==============

@graph_bp.route('/ontology/generate', methods=['POST'])
def generate_ontology():
    """
    接口1：上传文件，分析生成本体定义
    
    请求方式：multipart/form-data
    
    参数：
        files: 上传的文件（PDF/MD/TXT），可多个
        analysis_requirement: 分析需求描述（必填）
        project_name: 项目名称（可选）
        additional_context: 额外说明（可选）
        
    返回：
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "ontology": {
                    "entity_types": [...],
                    "edge_types": [...],
                    "analysis_summary": "..."
                },
                "files": [...],
                "total_text_length": 12345
            }
        }
    """
    try:
        logger.info("=== 开始生成本体定义 ===")
        
        # 获取参数 (兼容旧字段名 simulation_requirement)
        analysis_requirement = request.form.get('analysis_requirement', '') or request.form.get('simulation_requirement', '')
        project_name = request.form.get('project_name', 'Unnamed Project')
        additional_context = request.form.get('additional_context', '')

        logger.debug(f"项目名称: {project_name}")
        logger.debug(f"分析需求: {analysis_requirement[:100]}...")

        if not analysis_requirement:
            return jsonify({
                "success": False,
                "error": "请提供分析需求描述 (analysis_requirement)"
            }), 400
        
        # 获取上传的文件
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(not f.filename for f in uploaded_files):
            return jsonify({
                "success": False,
                "error": "请至少上传一个文档文件"
            }), 400
        
        # 创建项目
        project = ProjectManager.create_project(name=project_name)
        project.analysis_requirement = analysis_requirement
        logger.info(f"创建项目: {project.project_id}")
        
        # 保存文件并提取文本
        document_texts = []
        all_text = ""
        
        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                # 保存文件到项目目录
                file_info = ProjectManager.save_file_to_project(
                    project.project_id, 
                    file, 
                    file.filename
                )
                project.files.append({
                    "filename": file_info["original_filename"],
                    "size": file_info["size"]
                })
                
                # 提取文本
                text = FileParser.extract_text(file_info["path"])
                text = TextProcessor.preprocess_text(text)
                document_texts.append(text)
                all_text += f"\n\n=== {file_info['original_filename']} ===\n{text}"
        
        if not document_texts:
            ProjectManager.delete_project(project.project_id)
            return jsonify({
                "success": False,
                "error": "没有成功处理任何文档，请检查文件格式"
            }), 400
        
        # 保存提取的文本
        project.total_text_length = len(all_text)
        ProjectManager.save_extracted_text(project.project_id, all_text)
        logger.info(f"文本提取完成，共 {len(all_text)} 字符")
        
        # 生成本体
        logger.info("调用 LLM 生成本体定义...")
        generator = OntologyGenerator()
        ontology = generator.generate(
            document_texts=document_texts,
            analysis_requirement=analysis_requirement,
            additional_context=additional_context if additional_context else None
        )
        
        # 保存本体到项目
        entity_count = len(ontology.get("entity_types", []))
        edge_count = len(ontology.get("edge_types", []))
        logger.info(f"本体生成完成: {entity_count} 个实体类型, {edge_count} 个关系类型")
        
        project.ontology = {
            "entity_types": ontology.get("entity_types", []),
            "edge_types": ontology.get("edge_types", [])
        }
        project.analysis_summary = ontology.get("analysis_summary", "")
        project.status = ProjectStatus.ONTOLOGY_GENERATED
        ProjectManager.save_project(project)
        logger.info(f"=== 本体生成完成 === 项目ID: {project.project_id}")
        
        return jsonify({
            "success": True,
            "data": {
                "project_id": project.project_id,
                "project_name": project.name,
                "ontology": project.ontology,
                "analysis_summary": project.analysis_summary,
                "files": project.files,
                "total_text_length": project.total_text_length
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== 接口2：构建图谱 ==============

@graph_bp.route('/build', methods=['POST'])
def build_graph():
    """
    接口2：根据project_id构建图谱
    
    请求（JSON）：
        {
            "project_id": "proj_xxxx",  // 必填，来自接口1
            "graph_name": "图谱名称",    // 可选
            "chunk_size": 500,          // 可选，默认500
            "chunk_overlap": 50         // 可选，默认50
        }
        
    返回：
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "task_id": "task_xxxx",
                "message": "图谱构建任务已启动"
            }
        }
    """
    try:
        logger.info("=== 开始构建图谱 ===")
        
        # 检查配置（GraphRAG 不需要 ZEP_API_KEY，只需要 LLM_API_KEY）
        errors = []
        if not Config.LLM_API_KEY:
            errors.append("LLM_API_KEY未配置")
        if errors:
            logger.error(f"配置错误: {errors}")
            return jsonify({
                "success": False,
                "error": "配置错误: " + "; ".join(errors)
            }), 500
        
        # 解析请求
        data = request.get_json() or {}
        project_id = data.get('project_id')
        logger.debug(f"请求参数: project_id={project_id}")
        
        if not project_id:
            return jsonify({
                "success": False,
                "error": "请提供 project_id"
            }), 400
        
        # 获取项目
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": f"项目不存在: {project_id}"
            }), 404
        
        # 检查项目状态
        force = data.get('force', False)  # 强制重新构建
        
        if project.status == ProjectStatus.CREATED:
            return jsonify({
                "success": False,
                "error": "项目尚未生成本体，请先调用 /ontology/generate"
            }), 400
        
        if project.status == ProjectStatus.GRAPH_BUILDING and not force:
            return jsonify({
                "success": False,
                "error": "图谱正在构建中，请勿重复提交。如需强制重建，请添加 force: true",
                "task_id": project.graph_build_task_id
            }), 400
        
        # 如果强制重建，重置状态
        if force and project.status in [ProjectStatus.GRAPH_BUILDING, ProjectStatus.FAILED, ProjectStatus.GRAPH_COMPLETED]:
            project.status = ProjectStatus.ONTOLOGY_GENERATED
            project.graph_id = None
            project.graph_build_task_id = None
            project.error = None
        
        # 获取配置
        graph_name = data.get('graph_name', project.name or 'WeaverAgent Graph')
        chunk_size = data.get('chunk_size', project.chunk_size or Config.DEFAULT_CHUNK_SIZE)
        chunk_overlap = data.get('chunk_overlap', project.chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP)
        
        # 更新项目配置
        project.chunk_size = chunk_size
        project.chunk_overlap = chunk_overlap
        
        # 获取提取的文本
        text = ProjectManager.get_extracted_text(project_id)
        if not text:
            return jsonify({
                "success": False,
                "error": "未找到提取的文本内容"
            }), 400
        
        # 获取本体
        ontology = project.ontology
        if not ontology:
            return jsonify({
                "success": False,
                "error": "未找到本体定义"
            }), 400
        
        # 创建异步任务
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"构建图谱: {graph_name}")
        logger.info(f"创建图谱构建任务: task_id={task_id}, project_id={project_id}")
        
        # 更新项目状态
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        ProjectManager.save_project(project)
        
        # 启动后台任务
        def build_task():
            build_logger = get_logger('weaveragent.build')
            try:
                build_logger.info(f"[{task_id}] 开始构建图谱...")
                task_manager.update_task(
                    task_id, 
                    status=TaskStatus.PROCESSING,
                    message="初始化图谱构建服务..."
                )
                
                # 创建图谱构建服务
                builder = GraphBuilderService()  # GraphRAG 无需 api_key
                
                # 解析为完整文档（不预切碎，让 GraphRAG 按 token 切块）
                from ..services.graphrag_builder import _split_text_by_doc_markers
                task_manager.update_task(
                    task_id,
                    message="解析文档结构...",
                    progress=5
                )
                documents = _split_text_by_doc_markers(text)
                total_chunks = len(documents)
                build_logger.info(
                    f"[{task_id}] 准备写入 {total_chunks} 篇文档（共 {len(text):,} 字符），"
                    f"GraphRAG 将按 1200 token / chunk 自动切块"
                )

                # 创建图谱
                task_manager.update_task(
                    task_id,
                    message="创建GraphRAG图谱...",
                    progress=10
                )
                graph_id = builder.create_graph(name=graph_name)

                # 更新项目的graph_id
                project.graph_id = graph_id
                ProjectManager.save_project(project)

                # 设置本体
                task_manager.update_task(
                    task_id,
                    message="设置本体定义...",
                    progress=15
                )
                builder.set_ontology(graph_id, ontology)

                # 写入完整文档（progress_callback 签名是 (msg, progress_ratio)）
                def add_progress_callback(msg, progress_ratio):
                    progress = 15 + int(progress_ratio * 40)  # 15% - 55%
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress
                    )

                task_manager.update_task(
                    task_id,
                    message=f"开始写入 {total_chunks} 篇完整文档...",
                    progress=15
                )

                episode_uuids = builder.add_documents(
                    graph_id,
                    documents,
                    progress_callback=add_progress_callback
                )
                
                # 运行 GraphRAG 索引管道（实体/关系提取 + 社区检测 + 向量索引）
                task_manager.update_task(
                    task_id,
                    message="运行 GraphRAG 索引管道（LLM 推断中，可能需要几分钟）...",
                    progress=55
                )
                
                def indexing_progress_callback(msg, progress_ratio):
                    progress = 55 + int(progress_ratio * 35)  # 55% - 90%
                    task_manager.update_task(
                        task_id,
                        message=msg,
                        progress=progress
                    )
                
                builder._run_indexing_pipeline(graph_id, indexing_progress_callback)
                
                # 获取图谱数据
                task_manager.update_task(
                    task_id,
                    message="获取图谱数据...",
                    progress=95
                )
                graph_data = builder.get_graph_data(graph_id)
                
                # 更新项目状态
                project.status = ProjectStatus.GRAPH_COMPLETED
                ProjectManager.save_project(project)
                
                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)
                build_logger.info(f"[{task_id}] 图谱构建完成: graph_id={graph_id}, 节点={node_count}, 边={edge_count}")
                
                # 完成
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    message="图谱构建完成",
                    progress=100,
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "node_count": node_count,
                        "edge_count": edge_count,
                        "chunk_count": total_chunks
                    }
                )
                
            except Exception as e:
                # 更新项目状态为失败
                build_logger.error(f"[{task_id}] 图谱构建失败: {str(e)}")
                build_logger.debug(traceback.format_exc())
                
                project.status = ProjectStatus.FAILED
                project.error = str(e)
                ProjectManager.save_project(project)
                
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"构建失败: {str(e)}",
                    error=traceback.format_exc()
                )
        
        # 启动后台线程
        thread = threading.Thread(target=build_task, daemon=True)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "task_id": task_id,
                "message": "图谱构建任务已启动，请通过 /task/{task_id} 查询进度"
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== 任务查询接口 ==============

@graph_bp.route('/task/<task_id>', methods=['GET'])
def get_task(task_id: str):
    """
    查询任务状态
    """
    task = TaskManager().get_task(task_id)
    
    if not task:
        return jsonify({
            "success": False,
            "error": f"任务不存在: {task_id}"
        }), 404
    
    return jsonify({
        "success": True,
        "data": task.to_dict()
    })


@graph_bp.route('/tasks', methods=['GET'])
def list_tasks():
    """
    列出所有任务
    """
    tasks = TaskManager().list_tasks()
    
    return jsonify({
        "success": True,
        "data": [t.to_dict() for t in tasks],
        "count": len(tasks)
    })


# ============== 图谱数据接口 ==============

@graph_bp.route('/data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id: str):
    """
    获取图谱数据（节点和边）
    """
    try:
        builder = GraphBuilderService()
        graph_data = builder.get_graph_data(graph_id)
        
        return jsonify({
            "success": True,
            "data": graph_data
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@graph_bp.route('/delete/<graph_id>', methods=['DELETE'])
def delete_graph(graph_id: str):
    """
    删除图谱（GraphRAG 本地目录）
    """
    try:
        builder = GraphBuilderService()
        builder.delete_graph(graph_id)

        return jsonify({
            "success": True,
            "message": f"图谱已删除: {graph_id}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== 接口：追加文献到已有图谱 ==============

@graph_bp.route('/project/<project_id>/append-docs', methods=['POST'])
def append_documents(project_id: str):
    """
    向已构建完成的图谱追加新文献

    请求方式：multipart/form-data

    参数：
        files: 新增文献文件（PDF/MD/TXT），可多个
        chunk_size: 分块大小（可选，默认沿用项目配置）
        chunk_overlap: 分块重叠（可选，默认沿用项目配置）

    返回：
        {
            "success": true,
            "data": {
                "project_id": "proj_xxxx",
                "task_id": "task_xxxx",
                "message": "文献追加任务已启动"
            }
        }
    """
    try:
        logger.info(f"=== 开始追加文献: project_id={project_id} ===")

        # 获取项目
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({"success": False, "error": f"项目不存在: {project_id}"}), 404

        if not project.graph_id:
            return jsonify({
                "success": False,
                "error": "该项目尚未构建图谱，请先完成图谱构建再追加文献"
            }), 400

        if project.status == ProjectStatus.GRAPH_BUILDING:
            return jsonify({
                "success": False,
                "error": "图谱正在构建中，请等待构建完成后再追加文献"
            }), 400

        # 获取上传文件
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or all(not f.filename for f in uploaded_files):
            return jsonify({"success": False, "error": "请至少上传一个文献文件"}), 400

        # 获取分块参数
        chunk_size = request.form.get('chunk_size', type=int) or project.chunk_size or Config.DEFAULT_CHUNK_SIZE
        chunk_overlap = request.form.get('chunk_overlap', type=int) or project.chunk_overlap or Config.DEFAULT_CHUNK_OVERLAP

        # 保存新文件并提取文本
        new_texts = []
        new_file_infos = []

        for file in uploaded_files:
            if file and file.filename and allowed_file(file.filename):
                file_info = ProjectManager.save_file_to_project(
                    project.project_id,
                    file,
                    file.filename
                )
                text = FileParser.extract_text(file_info["path"])
                text = TextProcessor.preprocess_text(text)
                if text.strip():
                    new_texts.append(text)
                    new_file_infos.append({
                        "filename": file_info["original_filename"],
                        "size": file_info["size"]
                    })
                    logger.info(f"新文献提取成功: {file_info['original_filename']}, {len(text)} 字符")

        if not new_texts:
            return jsonify({
                "success": False,
                "error": "没有成功处理任何新文献，请检查文件格式"
            }), 400

        graph_id = project.graph_id

        # 创建追加任务
        task_manager = TaskManager()
        task_id = task_manager.create_task(f"追加文献到图谱: {project.name}")
        logger.info(f"创建追加任务: task_id={task_id}")

        # 立即更新项目文件列表和状态
        project.files.extend(new_file_infos)
        project.status = ProjectStatus.GRAPH_BUILDING
        project.graph_build_task_id = task_id
        project.error = None
        ProjectManager.save_project(project)

        def append_task():
            append_logger = get_logger('weaveragent.append')
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    message="初始化追加服务..."
                )

                builder = GraphBuilderService()  # GraphRAG 无需 api_key

                prev_data = builder.get_graph_data(graph_id)
                prev_nodes = prev_data.get("node_count", 0)
                prev_edges = prev_data.get("edge_count", 0)

                task_manager.update_task(
                    task_id, message="准备追加完整文档...", progress=5
                )
                # 整篇追加：每篇新文献作为一个完整文档写入（不预切碎）
                # 由 GraphRAG 按 token 自动切块（1200/chunk），避免句子被腰斩
                documents = [
                    {"name": info["filename"], "text": txt}
                    for info, txt in zip(new_file_infos, new_texts)
                    if txt.strip()
                ]
                total_chunks = len(documents)
                total_chars = sum(len(d["text"]) for d in documents)
                append_logger.info(
                    f"[{task_id}] 追加 {total_chunks} 篇文档（共 {total_chars:,} 字符），"
                    f"GraphRAG 将按 1200 token / chunk 自动切块"
                )

                def add_progress_callback(msg, progress_ratio):
                    progress = 10 + int(progress_ratio * 50)
                    task_manager.update_task(task_id, message=msg, progress=progress)

                task_manager.update_task(
                    task_id,
                    message=f"开始追加 {total_chunks} 篇文档...",
                    progress=10
                )

                episode_uuids = builder.add_documents(
                    graph_id,
                    documents,
                    progress_callback=add_progress_callback
                )

                task_manager.update_task(
                    task_id,
                    message="运行 GraphRAG 增量索引（LLM 推断中）...",
                    progress=60
                )

                def indexing_progress_callback(msg, progress_ratio):
                    progress = 60 + int(progress_ratio * 30)
                    task_manager.update_task(task_id, message=msg, progress=progress)

                builder._run_indexing_pipeline(graph_id, indexing_progress_callback, is_update_run=True)

                task_manager.update_task(task_id, message="获取图谱数据...", progress=95)
                graph_data = builder.get_graph_data(graph_id)

                project.status = ProjectStatus.GRAPH_COMPLETED
                project.error = None
                ProjectManager.save_project(project)

                node_count = graph_data.get("node_count", 0)
                edge_count = graph_data.get("edge_count", 0)
                new_nodes = node_count - prev_nodes
                new_edges = edge_count - prev_edges
                append_logger.info(
                    f"[{task_id}] 文献追加完成: 新增节点={new_nodes}, 新增边={new_edges}, "
                    f"总节点={node_count}, 总边={edge_count}"
                )

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.COMPLETED,
                    message="文献追加完成",
                    progress=100,
                    result={
                        "project_id": project_id,
                        "graph_id": graph_id,
                        "new_chunks": total_chunks,
                        "new_nodes": new_nodes,
                        "new_edges": new_edges,
                        "node_count": node_count,
                        "edge_count": edge_count,
                    }
                )

            except Exception as e:
                append_logger.error(f"[{task_id}] 文献追加失败: {str(e)}")
                append_logger.debug(traceback.format_exc())

                # 追加失败不影响已有图谱，恢复为已完成状态
                project.status = ProjectStatus.GRAPH_COMPLETED
                project.error = f"追加文献失败: {str(e)}"
                ProjectManager.save_project(project)

                task_manager.update_task(
                    task_id,
                    status=TaskStatus.FAILED,
                    message=f"追加失败: {str(e)}",
                    error=traceback.format_exc()
                )

        thread = threading.Thread(target=append_task, daemon=True)
        thread.start()

        return jsonify({
            "success": True,
            "data": {
                "project_id": project_id,
                "task_id": task_id,
                "new_files": [f["filename"] for f in new_file_infos],
                "message": "文献追加任务已启动，请通过 /api/graph/task/{task_id} 查询进度"
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
