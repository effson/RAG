const API_BASE = '/import-api'

/**
 * Upload files to the import service.
 * @param {File[]} files
 * @returns {Promise<{code: number, message: string, task_ids: string[]}>}
 */
export async function uploadFiles(files) {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData
  })

  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || '上传失败')
  }

  return res.json()
}

/**
 * Poll import task status.
 * @param {string} taskId
 * @returns {Promise<{code: number, task_id: string, status: string, done_list: string[], running_list: string[]}>}
 */
export async function getImportStatus(taskId) {
  const res = await fetch(`${API_BASE}/status/${taskId}`)
  if (!res.ok) {
    throw new Error('查询状态失败')
  }
  return res.json()
}
