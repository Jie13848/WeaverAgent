"""
GraphRAG 图谱记忆更新服务
使用 Microsoft GraphRAG 替代 Zep 实时记忆更新

完全兼容 zep_graph_memory_updater.py 的接口：
  AgentActivity, ZepGraphMemoryUpdater, ZepGraphMemoryManager

实现原理:
  Zep: 流式发送 episode → Zep Cloud 实时处理
  GraphRAG: 缓冲活动 → 批量写入文件 → 增量重建索引（is_update_run=True）

关键区别:
  - Zep 是即时的（毫秒级）
  - GraphRAG 是批量的（默认每5分钟触发一次增量索引）
  - 两种方案的检索结果在下次索引完成后才会更新
"""

import os
import uuid
import time
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from ..config import Config
from ..utils.logger import get_logger
from .graphrag_builder import get_graph_dir, GraphBuilderService

logger = get_logger('weaveragent.graphrag_memory_updater')


# ===========================================================
# 数据类（与 zep_graph_memory_updater.py 完全兼容）
# ===========================================================

@dataclass
class AgentActivity:
    """
    Agent 活动记录（与 Zep 版本完全兼容）

    to_episode_text() 生成的文本直接写入 GraphRAG input/ 目录。
    """
    platform: str           # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str        # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """将活动转换为自然语言文本（GraphRAG 直接摄入）"""
        action_descriptions = {
            "CREATE_POST":      self._describe_create_post,
            "LIKE_POST":        self._describe_like_post,
            "DISLIKE_POST":     self._describe_dislike_post,
            "REPOST":           self._describe_repost,
            "QUOTE_POST":       self._describe_quote_post,
            "FOLLOW":           self._describe_follow,
            "CREATE_COMMENT":   self._describe_create_comment,
            "LIKE_COMMENT":     self._describe_like_comment,
            "DISLIKE_COMMENT":  self._describe_dislike_comment,
            "SEARCH_POSTS":     self._describe_search,
            "SEARCH_USER":      self._describe_search_user,
            "MUTE":             self._describe_mute,
        }
        describe_func = action_descriptions.get(self.action_type, self._describe_generic)
        description = describe_func()
        return f"{self.agent_name}: {description}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        return f"发布了一条帖子：「{content}」" if content else "发布了一条帖子"

    def _describe_like_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author  = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f"点赞了{post_author}的帖子：「{post_content}」"
        elif post_content:
            return f"点赞了一条帖子：「{post_content}」"
        elif post_author:
            return f"点赞了{post_author}的一条帖子"
        return "点赞了一条帖子"

    def _describe_dislike_post(self) -> str:
        post_content = self.action_args.get("post_content", "")
        post_author  = self.action_args.get("post_author_name", "")
        if post_content and post_author:
            return f"踩了{post_author}的帖子：「{post_content}」"
        elif post_content:
            return f"踩了一条帖子：「{post_content}」"
        elif post_author:
            return f"踩了{post_author}的一条帖子"
        return "踩了一条帖子"

    def _describe_repost(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author  = self.action_args.get("original_author_name", "")
        if original_content and original_author:
            return f"转发了{original_author}的帖子：「{original_content}」"
        elif original_content:
            return f"转发了一条帖子：「{original_content}」"
        elif original_author:
            return f"转发了{original_author}的一条帖子"
        return "转发了一条帖子"

    def _describe_quote_post(self) -> str:
        original_content = self.action_args.get("original_content", "")
        original_author  = self.action_args.get("original_author_name", "")
        quote_content    = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        if original_content and original_author:
            base = f"引用了{original_author}的帖子「{original_content}」"
        elif original_content:
            base = f"引用了一条帖子「{original_content}」"
        elif original_author:
            base = f"引用了{original_author}的一条帖子"
        else:
            base = "引用了一条帖子"
        return base + (f"，并评论道：「{quote_content}」" if quote_content else "")

    def _describe_follow(self) -> str:
        target = self.action_args.get("target_user_name", "")
        return f"关注了用户「{target}」" if target else "关注了一个用户"

    def _describe_create_comment(self) -> str:
        content      = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author  = self.action_args.get("post_author_name", "")
        if content:
            if post_content and post_author:
                return f"在{post_author}的帖子「{post_content}」下评论道：「{content}」"
            elif post_content:
                return f"在帖子「{post_content}」下评论道：「{content}」"
            elif post_author:
                return f"在{post_author}的帖子下评论道：「{content}」"
            return f"评论道：「{content}」"
        return "发表了评论"

    def _describe_like_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author  = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f"点赞了{comment_author}的评论：「{comment_content}」"
        elif comment_content:
            return f"点赞了一条评论：「{comment_content}」"
        elif comment_author:
            return f"点赞了{comment_author}的一条评论"
        return "点赞了一条评论"

    def _describe_dislike_comment(self) -> str:
        comment_content = self.action_args.get("comment_content", "")
        comment_author  = self.action_args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f"踩了{comment_author}的评论：「{comment_content}」"
        elif comment_content:
            return f"踩了一条评论：「{comment_content}」"
        elif comment_author:
            return f"踩了{comment_author}的一条评论"
        return "踩了一条评论"

    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"搜索了「{query}」" if query else "进行了搜索"

    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f"搜索了用户「{query}」" if query else "搜索了用户"

    def _describe_mute(self) -> str:
        target = self.action_args.get("target_user_name", "")
        return f"屏蔽了用户「{target}」" if target else "屏蔽了一个用户"

    def _describe_generic(self) -> str:
        return f"执行了{self.action_type}操作"


# ===========================================================
# 图谱记忆更新器
# ===========================================================

class GraphRAGMemoryUpdater:
    """
    GraphRAG 图谱记忆更新器（替代 ZepGraphMemoryUpdater）

    工作模式:
    1. 活动通过 add_activity() 入队
    2. 后台线程按平台分组，累积到批次大小后写入 input/ 目录
    3. 达到重建阈值（默认 50 条活动）时，触发增量索引
    4. stop() 时强制刷新 + 触发最终增量索引
    """

    # 配置常量
    BATCH_SIZE           = 5        # 每批写入的活动数
    FLUSH_INTERVAL       = 2.0      # 队列检查间隔（秒）
    INDEX_THRESHOLD      = 50       # 触发增量索引的活动积累数
    INDEX_INTERVAL       = 300.0    # 最长增量索引间隔（秒），避免长时间不更新

    def __init__(self, project_id: str, graph_id: str):
        self.project_id  = project_id
        self.graph_id    = graph_id
        self.graph_dir   = get_graph_dir(graph_id)

        self._queue: Queue = Queue()
        self._running       = False
        self._worker_thread: Optional[threading.Thread] = None

        # 统计
        self._total_received  = 0
        self._total_written   = 0
        self._total_indexed   = 0
        self._start_time      = datetime.now()

        # 增量索引状态
        self._activities_since_last_index = 0
        self._last_index_time             = time.time()
        self._pending_platforms: Dict[str, List[AgentActivity]] = {}

        # 确保 input/ 目录存在
        (self.graph_dir / "input").mkdir(parents=True, exist_ok=True)

        logger.info(f"GraphRAG 记忆更新器已创建: project={project_id}, graph={graph_id}")

    def start(self):
        """启动后台写入线程"""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"graphrag-updater-{self.project_id}",
            daemon=True
        )
        self._worker_thread.start()
        logger.info(f"GraphRAG 记忆更新器已启动: project={self.project_id}")

    def stop(self):
        """停止更新器，刷新剩余活动并触发最终索引"""
        if not self._running:
            return
        self._running = False

        logger.info(f"正在停止记忆更新器: project={self.project_id}，刷新剩余活动...")

        # 等待队列排空
        try:
            self._queue.join()
        except Exception:
            pass

        # 强制写入所有待处理活动
        self._flush_all_pending()

        # 触发最终增量索引
        if self._activities_since_last_index > 0:
            self._run_incremental_index()

        if self._worker_thread:
            self._worker_thread.join(timeout=10)

        logger.info(
            f"记忆更新器已停止: project={self.project_id}, "
            f"共处理 {self._total_received} 条活动, "
            f"写入 {self._total_written} 个文件, "
            f"触发 {self._total_indexed} 次索引"
        )

    def add_activity(self, activity: AgentActivity):
        """添加一条活动记录（线程安全）"""
        self._total_received += 1
        self._queue.put(activity)

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """从 JSON 字典创建并添加活动（兼容原接口）"""
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", f"Agent_{data.get('agent_id', 0)}"),
            action_type=data.get("action_type", "UNKNOWN"),
            action_args=data.get("action_args", {}),
            round_num=data.get("round_num", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        self.add_activity(activity)

    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        elapsed = (datetime.now() - self._start_time).total_seconds()
        return {
            "project_id":       self.project_id,
            "graph_id":         self.graph_id,
            "total_received":   self._total_received,
            "total_written":    self._total_written,
            "total_indexed":    self._total_indexed,
            "queue_size":       self._queue.qsize(),
            "running":          self._running,
            "elapsed_seconds":  round(elapsed, 1),
            "activities_per_second": round(self._total_received / max(elapsed, 1), 2),
        }

    # ---------------------------------------------------------
    # 内部实现
    # ---------------------------------------------------------

    def _worker_loop(self):
        """后台工作线程主循环"""
        while self._running or not self._queue.empty():
            activities_batch: List[AgentActivity] = []

            # 收集一批活动
            try:
                activity = self._queue.get(timeout=self.FLUSH_INTERVAL)
                activities_batch.append(activity)

                # 非阻塞地继续取更多
                while len(activities_batch) < self.BATCH_SIZE:
                    try:
                        activity = self._queue.get_nowait()
                        activities_batch.append(activity)
                    except Empty:
                        break
            except Empty:
                pass

            # 写入文件
            if activities_batch:
                self._write_activities(activities_batch)
                for _ in activities_batch:
                    try:
                        self._queue.task_done()
                    except ValueError:
                        pass

            # 检查是否需要触发增量索引
            now = time.time()
            should_index = (
                self._activities_since_last_index >= self.INDEX_THRESHOLD
                or (
                    self._activities_since_last_index > 0
                    and (now - self._last_index_time) >= self.INDEX_INTERVAL
                )
            )
            if should_index:
                self._run_incremental_index()

    def _write_activities(self, activities: List[AgentActivity]):
        """将活动列表写入 input/ 目录的文本文件"""
        if not activities:
            return

        input_dir = self.graph_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        # 按平台分组合并
        by_platform: Dict[str, List[str]] = {}
        for act in activities:
            platform = act.platform or "general"
            if platform not in by_platform:
                by_platform[platform] = []
            by_platform[platform].append(act.to_episode_text())

        for platform, texts in by_platform.items():
            # 合并同平台活动为一个文本块
            combined_text = "\n".join(texts)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"activity_{platform}_{timestamp_str}_{uuid.uuid4().hex[:6]}.txt"

            try:
                with open(input_dir / filename, "w", encoding="utf-8") as f:
                    f.write(combined_text)

                self._total_written += len(texts)
                self._activities_since_last_index += len(texts)
                logger.debug(f"写入 {len(texts)} 条活动: {filename}")
            except Exception as e:
                logger.error(f"写入活动文件失败: {e}")

    def _flush_all_pending(self):
        """强制刷新所有队列中的待处理活动"""
        pending: List[AgentActivity] = []
        while not self._queue.empty():
            try:
                pending.append(self._queue.get_nowait())
            except Empty:
                break

        if pending:
            self._write_activities(pending)
            for _ in pending:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass
            logger.info(f"已强制刷新 {len(pending)} 条待处理活动")

    def _run_incremental_index(self):
        """触发增量 GraphRAG 索引"""
        logger.info(
            f"触发增量索引: project={self.project_id}, "
            f"积累活动数={self._activities_since_last_index}"
        )

        try:
            builder = GraphBuilderService()
            success = builder.run_incremental_update(self.graph_id)

            if success:
                self._total_indexed += 1
                logger.info(f"增量索引成功: project={self.project_id}")
            else:
                logger.warning(f"增量索引部分失败: project={self.project_id}")

        except Exception as e:
            logger.error(f"增量索引异常: {e}")
        finally:
            self._activities_since_last_index = 0
            self._last_index_time = time.time()


# ===========================================================
# 全局管理器（单例）
# ===========================================================

class GraphRAGMemoryManager:
    """
    GraphRAG 图谱记忆管理器（替代 ZepGraphMemoryManager）

    管理多个项目的记忆更新器实例，接口与原版完全兼容。
    """
    _instance: Optional["GraphRAGMemoryManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._updaters: Dict[str, GraphRAGMemoryUpdater] = {}
                    cls._instance._manager_lock = threading.Lock()
        return cls._instance

    def create_updater(
        self,
        project_id: str,
        graph_id: str
    ) -> GraphRAGMemoryUpdater:
        """创建并启动新的记忆更新器"""
        with self._manager_lock:
            # 若已存在，先停止旧的
            if project_id in self._updaters:
                old = self._updaters[project_id]
                logger.info(f"替换已有更新器: project={project_id}")
                threading.Thread(target=old.stop, daemon=True).start()

            updater = GraphRAGMemoryUpdater(project_id, graph_id)
            updater.start()
            self._updaters[project_id] = updater
            return updater

    def get_updater(self, project_id: str) -> Optional[GraphRAGMemoryUpdater]:
        """获取指定项目的更新器"""
        return self._updaters.get(project_id)

    def stop_updater(self, project_id: str):
        """停止指定项目的更新器"""
        with self._manager_lock:
            updater = self._updaters.pop(project_id, None)
            if updater:
                threading.Thread(target=updater.stop, daemon=True).start()
                logger.info(f"已停止更新器: project={project_id}")

    def stop_all(self):
        """停止所有更新器（应用退出时调用）"""
        with self._manager_lock:
            project_ids = list(self._updaters.keys())

        stop_threads = []
        for project_id in project_ids:
            with self._manager_lock:
                updater = self._updaters.pop(project_id, None)
            if updater:
                t = threading.Thread(target=updater.stop, daemon=True)
                t.start()
                stop_threads.append(t)

        for t in stop_threads:
            t.join(timeout=15)

        logger.info(f"已停止全部 {len(project_ids)} 个记忆更新器")

    def get_all_stats(self) -> Dict[str, Dict]:
        """获取所有更新器的统计信息"""
        return {
            pid: updater.get_stats()
            for pid, updater in self._updaters.items()
        }


# ===========================================================
# 向后兼容别名（确保旧代码无需修改）
# ===========================================================

# Zep 命名 → GraphRAG 实现
ZepGraphMemoryUpdater = GraphRAGMemoryUpdater
ZepGraphMemoryManager = GraphRAGMemoryManager
