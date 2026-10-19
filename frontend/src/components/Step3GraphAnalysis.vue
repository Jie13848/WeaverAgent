<template>
  <div class="graph-analysis-panel">
    <div class="scroll-container">

      <!-- Header Stats Row -->
      <div class="stats-overview">
        <div class="stat-card primary">
          <div class="stat-icon">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="3"></circle>
              <circle cx="4" cy="6" r="2"></circle>
              <circle cx="20" cy="6" r="2"></circle>
              <circle cx="4" cy="18" r="2"></circle>
              <circle cx="20" cy="18" r="2"></circle>
              <line x1="6" y1="6" x2="10" y2="10"></line>
              <line x1="18" y1="6" x2="14" y2="10"></line>
              <line x1="6" y1="18" x2="10" y2="14"></line>
              <line x1="18" y1="18" x2="14" y2="14"></line>
            </svg>
          </div>
          <div class="stat-body">
            <span class="stat-num">{{ totalNodes }}</span>
            <span class="stat-label">Knowledge Nodes</span>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="5" y1="12" x2="19" y2="12"></line>
              <polyline points="12 5 19 12 12 19"></polyline>
            </svg>
          </div>
          <div class="stat-body">
            <span class="stat-num">{{ totalEdges }}</span>
            <span class="stat-label">Relationships</span>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
            </svg>
          </div>
          <div class="stat-body">
            <span class="stat-num">{{ totalFiles }}</span>
            <span class="stat-label">Source Papers</span>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
            </svg>
          </div>
          <div class="stat-body">
            <span class="stat-num">{{ avgDegree }}</span>
            <span class="stat-label">Avg. Connections</span>
          </div>
        </div>
      </div>

      <!-- Node Type Distribution -->
      <div class="section-card" v-if="nodeTypeStats.length > 0">
        <div class="section-header">
          <span class="section-title">Node Type Distribution</span>
          <span class="section-badge">{{ nodeTypeStats.length }} types</span>
        </div>
        <div class="type-bars">
          <div
            v-for="item in nodeTypeStats"
            :key="item.type"
            class="type-bar-row"
          >
            <div class="type-label">
              <span class="type-dot" :style="{ background: typeColor(item.type) }"></span>
              <span class="type-name">{{ item.type }}</span>
            </div>
            <div class="bar-track">
              <div
                class="bar-fill"
                :style="{ width: item.percent + '%', background: typeColor(item.type) }"
              ></div>
            </div>
            <span class="type-count">{{ item.count }}</span>
            <span class="type-pct">{{ item.percent }}%</span>
          </div>
        </div>
      </div>

      <!-- Top Nodes by Connections (Hub Nodes) -->
      <div class="section-card" v-if="topNodes.length > 0">
        <div class="section-header">
          <span class="section-title">Hub Nodes · Most Connected</span>
          <span class="section-badge">Top {{ topNodes.length }}</span>
        </div>
        <div class="node-list">
          <div
            v-for="(node, idx) in topNodes"
            :key="node.uuid"
            class="node-item"
          >
            <span class="node-rank mono">#{{ idx + 1 }}</span>
            <div class="node-info">
              <div class="node-name-row">
                <span class="node-type-dot" :style="{ background: typeColor(node.primaryLabel) }"></span>
                <span class="node-name">{{ node.name }}</span>
                <span class="node-type-tag">{{ node.primaryLabel }}</span>
              </div>
              <p class="node-summary" v-if="node.summary">{{ node.summary.slice(0, 120) }}{{ node.summary.length > 120 ? '...' : '' }}</p>
            </div>
            <div class="node-degree">
              <span class="degree-num">{{ node.degree }}</span>
              <span class="degree-label">links</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Edge Type Distribution -->
      <div class="section-card" v-if="edgeTypeStats.length > 0">
        <div class="section-header">
          <span class="section-title">Relationship Type Distribution</span>
          <span class="section-badge">{{ edgeTypeStats.length }} types</span>
        </div>
        <div class="type-bars">
          <div
            v-for="item in edgeTypeStats"
            :key="item.type"
            class="type-bar-row"
          >
            <div class="type-label">
              <span class="type-dot" :style="{ background: edgeTypeColor(item.type) }"></span>
              <span class="type-name">{{ item.type }}</span>
            </div>
            <div class="bar-track">
              <div
                class="bar-fill"
                :style="{ width: item.percent + '%', background: edgeTypeColor(item.type) }"
              ></div>
            </div>
            <span class="type-count">{{ item.count }}</span>
            <span class="type-pct">{{ item.percent }}%</span>
          </div>
        </div>
      </div>

      <!-- Loading / Empty State -->
      <div v-if="!graphData && !loading" class="empty-state">
        <svg viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="#CCC" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <p>等待图谱构建完成...</p>
      </div>

      <div v-if="loading" class="loading-state">
        <div class="spinner"></div>
        <span>加载图谱统计数据...</span>
      </div>

      <!-- Action Footer -->
      <div class="action-footer" v-if="graphData">
        <button class="btn-ghost" @click="$emit('go-back')">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="15 18 9 12 15 6"></polyline>
          </svg>
          Back
        </button>
        <button class="btn-primary" @click="$emit('next-step')">
          Generate Report
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
        </button>
      </div>

    </div>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  graphData: { type: Object, default: null },
  projectData: { type: Object, default: null },
  loading: { type: Boolean, default: false },
})

defineEmits(['next-step', 'go-back'])

// ── 节点类型颜色映射 ──────────────────────────────────
const NODE_TYPE_COLORS = {
  Innovation: '#2563EB',
  PriorArt: '#7C3AED',
  Outcome: '#059669',
  Person: '#D97706',
  Organization: '#DC2626',
}
const EDGE_TYPE_COLORS = {
  BUILT_UPON: '#7C3AED',
  ACHIEVES: '#059669',
  IMPROVES_OVER: '#2563EB',
}
const FALLBACK_COLORS = ['#64748B', '#0891B2', '#BE185D', '#92400E']

const typeColor = (type) =>
  NODE_TYPE_COLORS[type] || FALLBACK_COLORS[type?.charCodeAt(0) % FALLBACK_COLORS.length] || '#64748B'

const edgeTypeColor = (type) =>
  EDGE_TYPE_COLORS[type] || FALLBACK_COLORS[type?.charCodeAt(0) % FALLBACK_COLORS.length] || '#64748B'

// ── 基础统计 ─────────────────────────────────────────
const totalNodes = computed(() => props.graphData?.nodes?.length ?? props.graphData?.node_count ?? 0)
const totalEdges = computed(() => props.graphData?.edges?.length ?? props.graphData?.edge_count ?? 0)
const totalFiles = computed(() => props.projectData?.files?.length ?? 0)

// 节点类型分布
const nodeTypeStats = computed(() => {
  const nodes = props.graphData?.nodes
  if (!nodes || nodes.length === 0) return []

  const counts = {}
  for (const node of nodes) {
    const labels = node.labels || []
    const primary = labels.find(l => !['Entity', 'Node'].includes(l)) || 'Unknown'
    counts[primary] = (counts[primary] || 0) + 1
  }

  const total = nodes.length
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([type, count]) => ({
      type,
      count,
      percent: Math.round((count / total) * 100),
    }))
})

// 边类型分布
const edgeTypeStats = computed(() => {
  const edges = props.graphData?.edges
  if (!edges || edges.length === 0) return []

  const counts = {}
  for (const edge of edges) {
    const type = edge.name || 'UNKNOWN'
    counts[type] = (counts[type] || 0) + 1
  }

  const total = edges.length
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([type, count]) => ({
      type,
      count,
      percent: Math.round((count / total) * 100),
    }))
})

// 平均连接度
const avgDegree = computed(() => {
  const nodes = props.graphData?.nodes
  const edges = props.graphData?.edges
  if (!nodes || nodes.length === 0) return '0'
  if (!edges || edges.length === 0) return '0'
  return ((edges.length * 2) / nodes.length).toFixed(1)
})

// Top Hub 节点（按连接度排序）
const topNodes = computed(() => {
  const nodes = props.graphData?.nodes
  const edges = props.graphData?.edges
  if (!nodes || nodes.length === 0) return []

  // 计算每个节点的度数
  const degreeMap = {}
  if (edges) {
    for (const edge of edges) {
      if (edge.source_uuid) degreeMap[edge.source_uuid] = (degreeMap[edge.source_uuid] || 0) + 1
      if (edge.target_uuid) degreeMap[edge.target_uuid] = (degreeMap[edge.target_uuid] || 0) + 1
      // 兼容不同字段名
      if (edge.source_node_uuid) degreeMap[edge.source_node_uuid] = (degreeMap[edge.source_node_uuid] || 0) + 1
      if (edge.target_node_uuid) degreeMap[edge.target_node_uuid] = (degreeMap[edge.target_node_uuid] || 0) + 1
    }
  }

  return nodes
    .map(n => ({
      ...n,
      primaryLabel: (n.labels || []).find(l => !['Entity', 'Node'].includes(l)) || 'Unknown',
      degree: degreeMap[n.uuid] || 0,
    }))
    .sort((a, b) => b.degree - a.degree)
    .slice(0, 10)
})
</script>

<style scoped>
.graph-analysis-panel {
  display: flex;
  flex-direction: column;
  font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif;
  background: #FAFAFA;
  padding: 24px;
  gap: 16px;
}

.scroll-container {
  display: contents;
}

/* ── Stats Overview ─────────────── */
.stats-overview {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.stat-card {
  background: #FFF;
  border: 1px solid #EAEAEA;
  border-radius: 10px;
  padding: 16px;
  display: flex;
  align-items: center;
  gap: 12px;
}

.stat-card.primary {
  border-color: #2563EB;
  background: #EFF6FF;
}

.stat-icon {
  width: 36px;
  height: 36px;
  background: #F5F5F5;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #555;
  flex-shrink: 0;
}

.stat-card.primary .stat-icon {
  background: #DBEAFE;
  color: #2563EB;
}

.stat-body {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.stat-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 22px;
  font-weight: 700;
  color: #111;
  line-height: 1;
}

.stat-label {
  font-size: 11px;
  color: #888;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* ── Section Cards ──────────────── */
.section-card {
  background: #FFF;
  border: 1px solid #EAEAEA;
  border-radius: 10px;
  overflow: hidden;
}

.section-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
  border-bottom: 1px solid #F0F0F0;
  background: #FAFAFA;
}

.section-title {
  font-size: 13px;
  font-weight: 700;
  color: #111;
  letter-spacing: 0.2px;
}

.section-badge {
  margin-left: auto;
  font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
  background: #F0F0F0;
  color: #666;
  padding: 2px 8px;
  border-radius: 4px;
}

/* ── Type Bars ──────────────────── */
.type-bars {
  padding: 16px 18px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.type-bar-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.type-label {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 140px;
  flex-shrink: 0;
}

.type-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.type-name {
  font-size: 12px;
  font-weight: 600;
  color: #333;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.bar-track {
  flex: 1;
  height: 6px;
  background: #F0F0F0;
  border-radius: 3px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
  opacity: 0.85;
}

.type-count {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 700;
  color: #333;
  width: 36px;
  text-align: right;
}

.type-pct {
  font-size: 11px;
  color: #999;
  width: 36px;
  text-align: right;
}

/* ── Node List ──────────────────── */
.node-list {
  padding: 8px 0;
}

.node-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 18px;
  border-bottom: 1px solid #F5F5F5;
  transition: background 0.15s;
}

.node-item:last-child {
  border-bottom: none;
}

.node-item:hover {
  background: #FAFAFA;
}

.node-rank {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: #BBB;
  width: 28px;
  flex-shrink: 0;
  padding-top: 2px;
}

.node-info {
  flex: 1;
  min-width: 0;
}

.node-name-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}

.node-type-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.node-name {
  font-size: 13px;
  font-weight: 700;
  color: #111;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 260px;
}

.node-type-tag {
  font-size: 10px;
  font-weight: 600;
  color: #888;
  background: #F0F0F0;
  padding: 1px 6px;
  border-radius: 3px;
  flex-shrink: 0;
}

.node-summary {
  font-size: 11px;
  color: #777;
  line-height: 1.5;
  margin: 0;
}

.node-degree {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1px;
  flex-shrink: 0;
}

.degree-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 18px;
  font-weight: 700;
  color: #333;
  line-height: 1;
}

.degree-label {
  font-size: 10px;
  color: #AAA;
  text-transform: uppercase;
}

/* ── Empty / Loading ────────────── */
.empty-state, .loading-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 60px 24px;
  color: #999;
  font-size: 13px;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #F0F0F0;
  border-top-color: #333;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* ── Action Footer ──────────────── */
.action-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 0 8px;
}

.btn-ghost {
  display: flex;
  align-items: center;
  gap: 6px;
  background: none;
  border: 1px solid #E0E0E0;
  color: #555;
  font-size: 13px;
  font-weight: 600;
  padding: 8px 16px;
  border-radius: 7px;
  cursor: pointer;
  transition: all 0.15s;
}

.btn-ghost:hover {
  border-color: #999;
  color: #111;
}

.btn-primary {
  display: flex;
  align-items: center;
  gap: 6px;
  background: #111;
  color: #FFF;
  border: none;
  font-size: 13px;
  font-weight: 600;
  padding: 9px 20px;
  border-radius: 7px;
  cursor: pointer;
  transition: all 0.15s;
}

.btn-primary:hover {
  background: #333;
}

.mono {
  font-family: 'JetBrains Mono', monospace;
}
</style>
