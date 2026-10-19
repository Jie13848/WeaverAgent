import service, { requestWithRetry } from './index'

/**
 * 生成本体（上传文档和分析需求）
 * @param {Object} data - 包含files, analysis_requirement, project_name等
 * @returns {Promise}
 */
export function generateOntology(formData) {
  return requestWithRetry(() => 
    service({
      url: '/api/graph/ontology/generate',
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  )
}

/**
 * 构建图谱
 * @param {Object} data - 包含project_id, graph_name等
 * @returns {Promise}
 */
export function buildGraph(data) {
  return requestWithRetry(() =>
    service({
      url: '/api/graph/build',
      method: 'post',
      data
    })
  )
}

/**
 * 查询任务状态
 * @param {String} taskId - 任务ID
 * @returns {Promise}
 */
export function getTaskStatus(taskId) {
  return service({
    url: `/api/graph/task/${taskId}`,
    method: 'get'
  })
}

/**
 * 获取图谱数据
 * @param {String} graphId - 图谱ID
 * @returns {Promise}
 */
export function getGraphData(graphId) {
  return service({
    url: `/api/graph/data/${graphId}`,
    method: 'get'
  })
}

/**
 * 获取项目信息
 * @param {String} projectId - 项目ID
 * @returns {Promise}
 */
export function getProject(projectId) {
  return service({
    url: `/api/graph/project/${projectId}`,
    method: 'get'
  })
}

/**
 * 获取项目列表（历史记录）
 * @param {Number} limit - 最大条数
 * @returns {Promise}
 */
export function listProjects(limit = 20) {
  return service({
    url: '/api/graph/project/list',
    method: 'get',
    params: { limit }
  })
}

/**
 * 重置项目状态（用于重新构建图谱）
 * @param {String} projectId - 项目ID
 * @returns {Promise}
 */
export function resetProject(projectId) {
  return service({
    url: `/api/graph/project/${projectId}/reset`,
    method: 'post'
  })
}

/**
 * 向已构建完成的图谱追加新文献
 * @param {String} projectId - 项目ID
 * @param {FormData} formData - 包含 files（多文件），可选 chunk_size / chunk_overlap
 * @returns {Promise}
 */
export function appendDocuments(projectId, formData) {
  return service({
    url: `/api/graph/project/${projectId}/append-docs`,
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}
