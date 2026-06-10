const API_BASE = '/query-api'

/**
 * Health check.
 * @returns {Promise<{ok: boolean}>}
 */
export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`)
  return res.json()
}

/**
 * Submit a query (streaming or non-streaming).
 * @param {string} query
 * @param {string} sessionId
 * @param {boolean} isStream
 * @returns {Promise<{message: string, session_id: string, answer?: string, done_list?: string[]}>}
 */
export async function submitQuery(query, sessionId, isStream) {
  const res = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      session_id: sessionId,
      is_stream: isStream
    })
  })

  if (!res.ok) {
    const msg = await res.text()
    throw new Error(msg || '请求失败')
  }

  return res.json()
}

/**
 * Create an EventSource for SSE streaming.
 * @param {string} sessionId
 * @returns {EventSource}
 */
export function createStreamSource(sessionId) {
  return new EventSource(`${API_BASE}/stream/${sessionId}`)
}

/**
 * Get chat history.
 * @param {string} sessionId
 * @param {number} limit
 * @returns {Promise<{session_id: string, items: Array}>}
 */
export async function getHistory(sessionId, limit = 50) {
  const res = await fetch(`${API_BASE}/history/${sessionId}?limit=${limit}`)
  if (!res.ok) return { items: [] }
  return res.json()
}

/**
 * Delete chat history.
 * @param {string} sessionId
 * @returns {Promise<{message: string, delete_count: number}>}
 */
export async function deleteHistory(sessionId) {
  const res = await fetch(`${API_BASE}/history/${sessionId}`, { method: 'DELETE' })
  return res.json()
}

/**
 * Poll query task status for non-streaming mode.
 * @param {string} taskId
 * @returns {Promise<{status: string, done_list: string[], running_list: string[]}>}
 */
export async function getQueryStatus(taskId) {
  const res = await fetch(`${API_BASE}/status/${taskId}`)
  if (!res.ok) throw new Error('查询状态失败')
  return res.json()
}
