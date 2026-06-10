/**
 * Escape HTML entities.
 */
export function escapeHtml(str) {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;')
}

/**
 * Format a Unix timestamp (seconds) to HH:MM.
 */
export function formatTime(ts) {
  if (!ts) return nowTime()
  const d = new Date(Number(ts) * 1000)
  if (Number.isNaN(d.getTime())) return nowTime()
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

/**
 * Get current time as HH:MM.
 */
export function nowTime() {
  const d = new Date()
  return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}

/**
 * Check if a URL points to an image.
 */
export function isImageUrl(url) {
  try {
    const u = new URL(url)
    return /\.(png|jpe?g|gif|webp|bmp|svg)$/i.test(u.pathname)
  } catch {
    return /\.(png|jpe?g|gif|webp|bmp|svg)(\?|#|$)/i.test(url || '')
  }
}

/**
 * Normalize a URL (encode spaces).
 */
export function normalizeUrl(rawUrl) {
  const s = String(rawUrl || '').trim()
  if (!s) return ''
  return s.replace(/\s/g, '%20')
}

/**
 * Extract all URLs from text (loose).
 */
export function extractUrls(text) {
  const s = String(text || '')
  const regex = /(https?:\/\/[^\s]+)/g
  const matches = s.match(regex) || []

  const trimTail = (u) => String(u || '').replace(/[)\]}'">，。,;\]】）＞]+$/g, '')
  const trimHead = (u) => String(u || '').replace(/^[<([{'"]+|^[＜（【\[]+/g, '')

  const seen = new Set()
  const urls = []
  for (const m of matches) {
    let u = trimHead(trimTail(m))
    if (u && !seen.has(u)) {
      seen.add(u)
      urls.push(u)
    }
  }
  return urls
}

/**
 * Find the last [图片] or 【图片】 marker index in text.
 */
export function findLastImageMarkerIndex(raw) {
  const s = String(raw || '')
  const re = /【\s*图片\s*】|\[\s*图片\s*\]/g
  let m
  let lastIdx = -1
  let lastLen = 0
  while ((m = re.exec(s)) !== null) {
    lastIdx = m.index
    lastLen = m[0].length
  }
  return { idx: lastIdx, len: lastLen }
}

/**
 * Parse answer text into text part and image URLs.
 * Images come after the last 【图片】 or [图片] marker.
 */
export function parseAnswerAndImages(text) {
  const raw = String(text || '')
  const { idx, len } = findLastImageMarkerIndex(raw)
  if (idx === -1) return { text: raw, images: [] }

  const before = raw.slice(0, idx).trimEnd()
  const after = raw.slice(idx + len).trim()
  const lines = after.split(/\r?\n/).map(l => l.trim()).filter(Boolean)

  const seen = new Set()
  const images = []
  for (const line of lines) {
    const urls = extractUrls(line)
    for (const u of urls) {
      const normalized = normalizeUrl(u)
      if (isImageUrl(normalized) && !seen.has(normalized)) {
        seen.add(normalized)
        images.push(normalized)
      }
    }
  }
  return { text: before, images }
}

/**
 * Check if answer text suggests showing images.
 */
export function shouldShowImages(answerText) {
  const t = String(answerText || '')
  const keywords = [
    '如图', '如下图', '见图', '见下图', '下图', '上图',
    '图片', '示意图', '结构图', '外观', '接线图', '电路图',
    '原理图', '安装图', '尺寸图', '截图'
  ]
  return keywords.some(k => t.includes(k))
}

/**
 * Collect all candidate image URLs from an answer and candidates list.
 */
export function collectImageUrls(answerText, candidateImageUrls) {
  const { images: fromBlock } = parseAnswerAndImages(answerText)
  const candidates = (Array.isArray(candidateImageUrls) ? candidateImageUrls : [])
    .map(normalizeUrl).filter(isImageUrl)
  const fromText = extractUrls(answerText).map(normalizeUrl).filter(isImageUrl)

  const all = new Set([...fromBlock, ...candidates, ...fromText])
  return Array.from(all)
}

/**
 * Deduplicate an array while preserving order.
 */
export function dedupeKeepOrder(arr) {
  const seen = new Set()
  const out = []
  for (const x of (Array.isArray(arr) ? arr : [])) {
    const v = String(x || '')
    if (!v || seen.has(v)) continue
    seen.add(v)
    out.push(v)
  }
  return out
}

/**
 * Format file size for display.
 */
export function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

/**
 * Generate a random session ID.
 */
export function generateSessionId() {
  return 'sess-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}

/**
 * Get or create a session ID from localStorage.
 */
export function getOrCreateSessionId() {
  let sid = localStorage.getItem('kb_session_id')
  if (!sid) {
    sid = generateSessionId()
    localStorage.setItem('kb_session_id', sid)
  }
  return sid
}
