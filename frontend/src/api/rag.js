/**
 * RAG（向量检索）API
 * 
 * 两阶段检索的第二阶段：在 Zep 图谱定位相关论文后，
 * 通过向量检索获取论文原文的细粒度段落。
 */

import service from './index'

/**
 * 构建/更新向量索引
 * @param {Object} options
 * @param {boolean} options.force_rebuild - 强制重建（忽略已有索引）
 * @param {string[]} options.files       - 只索引指定文件（空则扫描全部）
 */
export function buildRagIndex(options = {}) {
  return service.post('/api/rag/index', options)
}

/**
 * 查询索引状态
 */
export function getRagStatus() {
  return service.get('/api/rag/status')
}

/**
 * 向量检索论文段落
 * @param {Object} params
 * @param {string}   params.query             - 检索问题（必填）
 * @param {number}   params.top_k             - 返回段落数（默认 6）
 * @param {string[]} params.filter_filenames  - 限定检索的文件名列表（可选）
 */
export function ragSearch(params) {
  return service.post('/api/rag/search', params)
}
