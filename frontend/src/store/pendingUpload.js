/**
 * 临时存储待上传的文件和需求
 * 用于首页点击启动引擎后立即跳转，在Process页面再进行API调用
 */
import { reactive } from 'vue'

const state = reactive({
  files: [],
  analysisRequirement: '',
  isPending: false
})

export function setPendingUpload(files, requirement) {
  state.files = files
  state.analysisRequirement = requirement
  state.isPending = true
}

export function getPendingUpload() {
  return {
    files: state.files,
    analysisRequirement: state.analysisRequirement,
    isPending: state.isPending
  }
}

export function clearPendingUpload() {
  state.files = []
  state.analysisRequirement = ''
  state.isPending = false
}

export default state
