import { reactive } from 'vue'
import { uploadFiles, getImportStatus } from '../api/import'
import { formatFileSize } from '../utils/helpers'

/**
 * 模块级上传状态存储 — 独立于组件生命周期，
 * 切换路由后回来仍能看到上传进度。
 */
const files = reactive([])

/** @type {Record<string, number>} */
const pollTimers = {}

function findById(id) {
  return files.find(f => f.id === id)
}

function startUpload(file) {
  const id = 'f-' + Math.random().toString(36).slice(2, 9)
  const entry = {
    id,
    name: file.name,
    size: file.size,
    sizeFormatted: formatFileSize(file.size),
    file,
    taskId: '',
    status: 'idle',
    progress: 0,
    doneList: [],
    runningList: []
  }
  files.unshift(entry)

  // 从 reactive 数组中取出 Proxy
  const proxy = files.find(f => f.id === id)
  _doUpload(proxy)
}

async function _doUpload(entry) {
  entry.status = 'uploading'
  entry.progress = 30

  try {
    const result = await uploadFiles([entry.file])
    entry.taskId = result.task_ids?.[0] || ''
    entry.status = 'processing'
    entry.progress = 30
    _startPoll(entry)
  } catch (e) {
    console.error('Upload failed:', e)
    entry.status = 'error'
  }
}

function _startPoll(entry) {
  // 避免同一个 entry 重复轮询
  if (pollTimers[entry.id]) clearInterval(pollTimers[entry.id])

  const timer = setInterval(async () => {
    try {
      const data = await getImportStatus(entry.taskId)
      entry.doneList = data.done_list || []
      entry.runningList = data.running_list || []

      if (data.status === 'completed') {
        entry.status = 'completed'
        entry.progress = 100
        clearInterval(timer)
        delete pollTimers[entry.id]
      } else if (data.status === 'processing') {
        entry.status = 'processing'
      } else if (data.status === 'failed') {
        entry.status = 'error'
        clearInterval(timer)
        delete pollTimers[entry.id]
      }
    } catch (e) {
      console.error('Poll error:', e)
    }
  }, 2000)

  pollTimers[entry.id] = timer
}

function clearFiles() {
  // 停止所有轮询
  Object.values(pollTimers).forEach(clearInterval)
  for (const key of Object.keys(pollTimers)) {
    delete pollTimers[key]
  }
  files.splice(0, files.length)
}

/**
 * 清理已完成的轮询定时器（组件卸载时可调用，避免浪费）
 */
function cleanupCompletedTimers() {
  for (const [id, timer] of Object.entries(pollTimers)) {
    const entry = findById(id)
    if (!entry || entry.status === 'completed' || entry.status === 'error') {
      clearInterval(timer)
      delete pollTimers[id]
    }
  }
}

export function useImportStore() {
  return {
    files,
    startUpload,
    clearFiles,
    cleanupCompletedTimers
  }
}
