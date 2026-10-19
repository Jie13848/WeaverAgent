<template>
  <div class="env-setup-panel">
    <div class="scroll-container">

      <!-- 项目信息卡 -->
      <div class="info-card">
        <div class="card-header">
          <span class="card-icon">◆</span>
          <span class="card-title">项目信息</span>
        </div>
        <div class="info-rows">
          <div class="info-row">
            <span class="info-label">Project ID</span>
            <span class="info-value mono">{{ projectData?.project_id || '—' }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">Graph ID</span>
            <span class="info-value mono">{{ projectData?.graph_id || '—' }}</span>
          </div>
          <div class="info-row">
            <span class="info-label">分析需求</span>
            <span class="info-value req-text">{{ projectData?.analysis_requirement || '—' }}</span>
          </div>
        </div>
      </div>

      <!-- 图谱统计 -->
      <div class="stats-card" v-if="graphData">
        <div class="card-header">
          <span class="card-icon">◈</span>
          <span class="card-title">知识图谱概览</span>
          <span class="badge-ready">已就绪</span>
        </div>
        <div class="stats-grid">
          <div class="stat-item">
            <span class="stat-number">{{ totalNodes }}</span>
            <span class="stat-label">节点总数</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">{{ totalEdges }}</span>
            <span class="stat-label">关系总数</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">{{ nodeTypeCount }}</span>
            <span class="stat-label">节点类型</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">{{ edgeTypeCount }}</span>
            <span class="stat-label">关系类型</span>
          </div>
        </div>

        <!-- 节点类型分布 -->
        <div class="type-distribution" v-if="nodeTypes.length">
          <div class="dist-title">节点类型分布</div>
          <div class="dist-rows">
            <div v-for="t in nodeTypes" :key="t.type" class="dist-row">
              <span class="type-dot" :style="{ background: typeColor(t.type) }"></span>
              <span class="type-name">{{ t.type }}</span>
              <div class="type-bar-wrap">
                <div class="type-bar" :style="{ width: barWidth(t.count) + '%', background: typeColor(t.type) }"></div>
              </div>
              <span class="type-count">{{ t.count }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 无图谱提示 -->
      <div class="empty-card" v-else>
        <div class="empty-icon">⚠</div>
        <p>尚未构建知识图谱，请返回 Step 1 完成图谱构建。</p>
      </div>

      <!-- Top 节点预览 -->
      <div class="nodes-card" v-if="topNodes.length">
        <div class="card-header">
          <span class="card-icon">⬡</span>
          <span class="card-title">核心节点预览（Top {{ topNodes.length }}）</span>
        </div>
        <div class="nodes-list">
          <div v-for="(node, idx) in topNodes" :key="idx" class="node-row">
            <span class="node-rank">{{ String(idx + 1).padStart(2, '0') }}</span>
            <span class="node-dot" :style="{ background: typeColor(node.type) }"></span>
            <span class="node-name">{{ node.name }}</span>
            <span class="node-type-tag" :style="{ color: typeColor(node.type) }">{{ node.type }}</span>
          </div>
        </div>
      </div>

      <!-- 底部操作栏 -->
      <div class="action-bar">
        <button class="btn-back" @click="$emit('go-back')">← 返回</button>
        <div class="action-right">
          <!-- 追加文献按钮（仅图谱已就绪时显示） -->
          <button
            v-if="graphData"
            class="btn-append"
            :disabled="appending"
            @click="triggerAppend"
          >
            <span v-if="!appending">＋ 添加文献</span>
            <span v-else>追加中…</span>
          </button>
          <input
            ref="appendFileInput"
            type="file"
            multiple
            accept=".pdf,.md,.txt"
            style="display:none"
            @change="onAppendFilesSelected"
          />
          <button
            class="btn-next"
            :disabled="!graphData"
            @click="$emit('next-step')"
          >
            进入图谱分析 →
          </button>
        </div>
      </div>

      <!-- 追加进度面板 -->
      <div v-if="appendStatus" class="append-progress-card" :class="appendStatus.type">
        <div class="append-progress-header">
          <span class="append-progress-icon">
            <span v-if="appendStatus.type === 'info'" class="spinner-inline"></span>
            <span v-else-if="appendStatus.type === 'success'">✓</span>
            <span v-else>✕</span>
          </span>
          <span class="append-progress-msg">{{ appendStatus.message }}</span>
        </div>
        <div v-if="appendStatus.type === 'info' && appendPercent > 0" class="append-bar-wrap">
          <div class="append-bar-track">
            <div class="append-bar-fill" :style="{ width: appendPercent + '%' }"></div>
          </div>
          <span class="append-bar-pct">{{ appendPercent }}%</span>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { appendDocuments } from '../api/graph'
import { getTaskStatus } from '../api/graph'

const props = defineProps({
  projectData: Object,
  graphData: Object,
  systemLogs: Array
})

const emit = defineEmits(['go-back', 'next-step', 'add-log', 'graph-updated'])

// ── 追加文献 ──
const appendFileInput = ref(null)
const appending = ref(false)
const appendStatus = ref(null)
const appendPercent = ref(0)

function triggerAppend() {
  appendFileInput.value?.click()
}

async function onAppendFilesSelected(event) {
  const files = Array.from(event.target.files)
  if (!files.length) return

  event.target.value = ''

  if (!props.projectData?.project_id) {
    appendStatus.value = { type: 'error', message: '缺少 project_id，无法追加' }
    return
  }

  appending.value = true
  appendPercent.value = 0
  appendStatus.value = { type: 'info', message: `正在上传 ${files.length} 个文件…` }

  try {
    const formData = new FormData()
    files.forEach(f => formData.append('files', f))

    const res = await appendDocuments(props.projectData.project_id, formData)
    if (!res.success) {
      throw new Error(res.error || '追加请求失败')
    }

    const taskId = res.data.task_id
    const fileNames = res.data.new_files?.join('、') || ''
    appendStatus.value = { type: 'info', message: `文件已上传（${fileNames}），正在索引到图谱…` }
    emit('add-log', `[追加文献] 任务已启动: ${taskId}`)

    await pollAppendTask(taskId)

  } catch (err) {
    appendStatus.value = { type: 'error', message: `追加失败：${err.message}` }
    emit('add-log', `[追加文献] 失败: ${err.message}`)
  } finally {
    appending.value = false
  }
}

async function pollAppendTask(taskId) {
  const POLL_INTERVAL = 2500
  let consecutiveErrors = 0

  while (true) {
    await new Promise(r => setTimeout(r, POLL_INTERVAL))
    try {
      const res = await getTaskStatus(taskId)
      const task = res.data
      if (!task) { consecutiveErrors++; continue }

      consecutiveErrors = 0
      const progress = task.progress ?? 0
      const msg = task.message || ''
      appendPercent.value = progress

      if (task.status === 'completed') {
        const nc = task.result?.node_count ?? task.result?.graph_info?.node_count ?? '—'
        const ec = task.result?.edge_count ?? task.result?.graph_info?.edge_count ?? '—'
        const chunks = task.result?.new_chunks ?? task.result?.chunks_appended ?? '—'
        const nodesAdded = task.result?.new_nodes ?? task.result?.nodes_added ?? 0
        const edgesAdded = task.result?.new_edges ?? task.result?.edges_added ?? 0
        appendPercent.value = 100
        appendStatus.value = {
          type: 'success',
          message: `追加完成 — 新增 ${nodesAdded} 节点 / ${edgesAdded} 关系，图谱共 ${nc} 节点 / ${ec} 关系（${chunks} 个文本块）`
        }
        emit('add-log', `[追加文献] 完成: 新增节点=${nodesAdded}, 新增关系=${edgesAdded}, 总节点=${nc}`)
        emit('graph-updated')
        return
      }

      if (task.status === 'failed') {
        throw new Error(task.error || task.message || '追加失败')
      }

      appendStatus.value = { type: 'info', message: msg || `索引中 ${progress}%` }
    } catch (err) {
      if (err.message.includes('追加失败') || err.message.includes('failed')) {
        appendStatus.value = { type: 'error', message: `追加失败：${err.message}` }
        emit('add-log', `[追加文献] 失败: ${err.message}`)
        return
      }
      consecutiveErrors++
      if (consecutiveErrors > 20) {
        appendStatus.value = { type: 'error', message: '无法连接后端，请检查服务是否正常运行' }
        emit('add-log', '[追加文献] 轮询失败：连续多次无法获取任务状态')
        return
      }
    }
  }
}


const totalNodes = computed(() => {
  if (!props.graphData) return 0
  return props.graphData.node_count ?? props.graphData.nodes?.length ?? 0
})

const totalEdges = computed(() => {
  if (!props.graphData) return 0
  return props.graphData.edge_count ?? props.graphData.edges?.length ?? 0
})

const nodeTypes = computed(() => {
  if (!props.graphData?.nodes?.length) return []
  const counter = {}
  for (const n of props.graphData.nodes) {
    const t = n.type || n.label || 'Unknown'
    counter[t] = (counter[t] || 0) + 1
  }
  return Object.entries(counter)
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count)
})

const edgeTypes = computed(() => {
  if (!props.graphData?.edges?.length) return []
  const counter = {}
  for (const e of props.graphData.edges) {
    const t = e.type || e.relation || e.label || 'related'
    counter[t] = (counter[t] || 0) + 1
  }
  return Object.entries(counter).map(([type, count]) => ({ type, count }))
})

const nodeTypeCount = computed(() => nodeTypes.value.length)
const edgeTypeCount = computed(() => edgeTypes.value.length)

const topNodes = computed(() => {
  if (!props.graphData?.nodes?.length) return []
  const nodes = props.graphData.nodes
  const edges = props.graphData.edges || []
  const degree = {}
  for (const e of edges) {
    const s = e.source ?? e.from
    const t = e.target ?? e.to
    if (s) degree[s] = (degree[s] || 0) + 1
    if (t) degree[t] = (degree[t] || 0) + 1
  }
  return [...nodes]
    .sort((a, b) => (degree[b.id] || 0) - (degree[a.id] || 0))
    .slice(0, 10)
    .map(n => ({
      name: n.name || n.label || n.id,
      type: n.type || n.label || 'Unknown'
    }))
})

const TYPE_COLORS = {
  paper: '#2563EB',
  method: '#7C3AED',
  innovation: '#059669',
  task: '#D97706',
  dataset: '#0891B2',
  metric: '#DC2626',
  baseline: '#9333EA',
  author: '#EA580C',
  organization: '#4F46E5',
  concept: '#0D9488',
  variable: '#64748B',
  parameter: '#78716C',
}

const typeColor = (type) => TYPE_COLORS[(type || '').toLowerCase()] ?? '#6B7280'
const maxNodeCount = computed(() => nodeTypes.value[0]?.count || 1)
const barWidth = (count) => Math.round((count / maxNodeCount.value) * 100)
</script>

<style scoped>
.env-setup-panel {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #F8F9FA;
  font-family: 'Inter', 'Noto Sans SC', system-ui, sans-serif;
  overflow: hidden;
}
.scroll-container {
  flex: 1;
  overflow-y: auto;
  padding: 24px 28px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.info-card, .stats-card, .nodes-card, .empty-card {
  background: #fff;
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 20px 24px;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}
.card-icon { font-size: 14px; color: #111; }
.card-title { font-size: 14px; font-weight: 600; color: #111; letter-spacing: 0.02em; }
.badge-ready {
  margin-left: auto;
  font-size: 11px;
  font-weight: 600;
  color: #059669;
  background: #D1FAE5;
  padding: 2px 8px;
  border-radius: 20px;
}
.info-rows { display: flex; flex-direction: column; gap: 10px; }
.info-row { display: flex; align-items: flex-start; gap: 12px; }
.info-label {
  font-size: 11px;
  font-weight: 600;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  min-width: 90px;
  padding-top: 1px;
}
.info-value { font-size: 13px; color: #374151; flex: 1; word-break: break-all; }
.info-value.mono { font-family: 'JetBrains Mono', 'SF Mono', monospace; font-size: 12px; color: #1F2937; }
.info-value.req-text { line-height: 1.6; }
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 20px;
}
.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: #F9FAFB;
  border: 1px solid #F3F4F6;
  border-radius: 8px;
  padding: 12px 8px;
  gap: 4px;
}
.stat-number { font-size: 22px; font-weight: 700; color: #111; line-height: 1; }
.stat-label { font-size: 11px; color: #9CA3AF; text-align: center; }
.dist-title {
  font-size: 12px;
  font-weight: 600;
  color: #6B7280;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-bottom: 10px;
}
.dist-rows { display: flex; flex-direction: column; gap: 8px; }
.dist-row { display: flex; align-items: center; gap: 8px; }
.type-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.type-name { font-size: 12px; color: #374151; width: 110px; flex-shrink: 0; }
.type-bar-wrap { flex: 1; height: 6px; background: #F3F4F6; border-radius: 3px; overflow: hidden; }
.type-bar { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
.type-count { font-size: 12px; font-weight: 600; color: #374151; min-width: 30px; text-align: right; }
.nodes-list { display: flex; flex-direction: column; gap: 6px; }
.node-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  background: #F9FAFB;
  border-radius: 6px;
}
.node-rank { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #9CA3AF; min-width: 22px; }
.node-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.node-name { font-size: 13px; color: #1F2937; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.node-type-tag { font-size: 11px; font-weight: 600; flex-shrink: 0; }
.empty-card { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 40px; color: #9CA3AF; }
.empty-icon { font-size: 28px; }
.empty-card p { font-size: 13px; text-align: center; line-height: 1.6; }
.action-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0 4px;
}
.action-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.btn-back {
  padding: 10px 20px;
  border: 1px solid #D1D5DB;
  border-radius: 8px;
  background: #fff;
  color: #374151;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-back:hover { background: #F9FAFB; }
.btn-append {
  padding: 10px 20px;
  border: 1px solid #2563EB;
  border-radius: 8px;
  background: #EFF6FF;
  color: #2563EB;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
}
.btn-append:hover:not(:disabled) { background: #DBEAFE; }
.btn-append:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-next {
  padding: 10px 28px;
  border: none;
  border-radius: 8px;
  background: #111;
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
}
.btn-next:disabled { opacity: 0.35; cursor: not-allowed; }
.btn-next:not(:disabled):hover { opacity: 0.85; }
/* ── Append Progress Card ── */
.append-progress-card {
  padding: 16px 20px;
  border-radius: 10px;
  font-size: 13px;
  line-height: 1.5;
}
.append-progress-card.info {
  background: #EFF6FF;
  color: #1D4ED8;
  border: 1px solid #BFDBFE;
}
.append-progress-card.success {
  background: #D1FAE5;
  color: #065F46;
  border: 1px solid #6EE7B7;
}
.append-progress-card.error {
  background: #FEF2F2;
  color: #991B1B;
  border: 1px solid #FECACA;
}
.append-progress-header {
  display: flex;
  align-items: center;
  gap: 10px;
}
.append-progress-icon {
  flex-shrink: 0;
  font-size: 14px;
  font-weight: 700;
}
.append-progress-msg {
  flex: 1;
  font-weight: 500;
}
.append-bar-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 12px;
}
.append-bar-track {
  flex: 1;
  height: 8px;
  background: rgba(59, 130, 246, 0.15);
  border-radius: 4px;
  overflow: hidden;
}
.append-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #3B82F6, #60A5FA);
  border-radius: 4px;
  transition: width 0.6s ease;
  min-width: 2px;
}
.append-bar-pct {
  font-size: 12px;
  font-weight: 700;
  font-family: 'JetBrains Mono', 'SF Mono', monospace;
  color: #2563EB;
  min-width: 36px;
  text-align: right;
}

/* Inline spinner for append */
.spinner-inline {
  display: inline-block;
  width: 14px;
  height: 14px;
  border: 2px solid rgba(59, 130, 246, 0.3);
  border-top-color: #3B82F6;
  border-radius: 50%;
  animation: spin-inline 0.8s linear infinite;
}
@keyframes spin-inline {
  to { transform: rotate(360deg); }
}
</style>
