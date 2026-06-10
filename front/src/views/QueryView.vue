<template>
  <div class="query-page">
    <!-- Fixed Toolbar -->
    <div class="chat-toolbar">
      <label class="toggle-label">
        <input type="checkbox" v-model="isStream" />
        <span>流式输出</span>
      </label>
      <button class="btn-clear" @click="onClearHistory">清空历史</button>
    </div>
    <!-- Messages Area -->
    <div class="chat-body" ref="chatBodyRef">
        <div class="dayline">今天</div>

        <!-- Welcome -->
        <div class="msg bot">
          <div class="avatar bot-avatar"><BotAvatar /></div>
          <div class="msg-content">
            <div class="bubble">
              你好，我是基石智库知识库客服。你可以直接提问，我会在"阶段进度"里展示处理过程。
            </div>
            <div class="meta">提示：可以问"如何使用万用表测量电压？"</div>
          </div>
        </div>

        <!-- History messages -->
        <template v-for="msg in messages" :key="msg._id || msg.clientId">
          <!-- User message -->
          <div v-if="msg.role === 'user'" class="msg user">
            <div class="msg-content">
              <div class="bubble">{{ msg.text }}</div>
              <div class="meta">{{ formatTime(msg.ts) }}</div>
            </div>
            <div class="avatar user-avatar"><UserAvatar /></div>
          </div>

          <!-- Bot message (history) -->
          <div v-else class="msg bot">
            <div class="avatar bot-avatar"><BotAvatar /></div>
            <div class="msg-content">
              <div class="bubble">
                <AnswerDisplay :text="msg.text || ''" :image-urls="msg.image_urls || []" />
              </div>
              <div class="meta">{{ formatTime(msg.ts) }}</div>
            </div>
          </div>
        </template>

        <!-- Active streaming bot message -->
        <div v-if="activeMsg" class="msg bot" :key="activeMsg.clientId">
          <div class="avatar bot-avatar"><BotAvatar /></div>
          <div class="msg-content">
            <div class="bubble">
              <!-- Typing indicator when no content yet -->
              <div v-if="!activeMsg.hasContent && activeMsg.status !== 'completed' && activeMsg.status !== 'failed'" class="typing">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
              </div>

              <!-- Answer text during streaming -->
              <AnswerDisplay
                v-if="activeMsg.hasContent || activeMsg.answerText"
                :text="activeMsg.answerText || ''"
                :image-urls="activeMsg.imageUrls || []"
              />

              <!-- Progress details -->
              <details class="progress-details" :open="activeMsg.status !== 'completed' && activeMsg.status !== 'failed'">
                <summary>
                  阶段进度（已完成 {{ activeMsg.doneList.length }}，进行中 {{ activeMsg.runningList.length }}，状态：{{ progressStatusMap[activeMsg.status] || activeMsg.status || '等待中' }}）
                </summary>
                <ul class="progress-list">
                  <li v-if="activeMsg.doneList.length === 0 && activeMsg.runningList.length === 0">暂无进度</li>
                  <li v-for="item in activeMsg.doneList" :key="'d-' + item"><span class="log-check">✓</span> {{ item }}已完成</li>
                  <li v-for="item in activeMsg.runningList" :key="'r-' + item"><span v-if="activeMsg.status === 'failed'" class="log-cross">✗</span><span v-else class="spinner"></span>正在进行{{ item }}...</li>
                </ul>
              </details>

              <!-- Error display -->
              <div v-if="activeMsg.status === 'failed'" class="error-msg">
                {{ activeMsg.error || '请求失败' }}
              </div>
            </div>
            <div class="meta">{{ activeMsg.time }}</div>
          </div>
        </div>
      </div>

      <!-- Composer -->
      <div class="composer">
        <textarea
          ref="inputRef"
          v-model="inputText"
          placeholder="请输入问题（Enter 发送，Shift+Enter 换行）"
          @keydown="onInputKeydown"
          :disabled="isSending"
          rows="1"
        ></textarea>
        <button
          class="btn btn-primary send-btn"
          :disabled="isSending || !inputText.trim()"
          @click="onSend"
        >
          发送
        </button>
      </div>
  </div>
</template>

<script setup>
import { ref, reactive, nextTick, onMounted, onUnmounted, watch } from 'vue'
import {
  submitQuery,
  createStreamSource,
  getHistory,
  deleteHistory
} from '../api/query'
import {
  formatTime,
  nowTime,
  getOrCreateSessionId,
  collectImageUrls
} from '../utils/helpers'
import AnswerDisplay from '../components/AnswerDisplay.vue'
import BotAvatar from '../components/BotAvatar.vue'
import UserAvatar from '../components/UserAvatar.vue'

const chatBodyRef = ref(null)
const inputRef = ref(null)
const inputText = ref('')
const isSending = ref(false)
const isStream = ref(true)
const sessionId = getOrCreateSessionId()

/** @type {Array} history messages */
const messages = ref([])

/** @type {Ref<{clientId: string, hasContent: boolean, answerText: string, status: string, doneList: string[], runningList: string[], imageUrls: string[], error: string, time: string} | null>} */
const activeMsg = ref(null)

const progressStatusMap = {
  pending: '等待中',
  processing: '处理中',
  completed: '已完成',
  failed: '失败'
}

// ----- Lifecycle -----
onMounted(async () => {
  await loadHistory()
  nextTick(() => inputRef.value?.focus())
})

// ----- History -----
async function loadHistory() {
  try {
    const data = await getHistory(sessionId, 50)
    const items = Array.isArray(data.items) ? data.items : []
    if (items.length > 0) {
      messages.value = items
      nextTick(scrollToBottom)
    }
  } catch {
    // silent
  }
}

async function onClearHistory() {
  if (!confirm('确定要清空当前会话的历史记录吗？这将无法恢复。')) return
  try {
    await deleteHistory(sessionId)
  } catch (e) {
    console.error('Clear history failed:', e)
    alert('服务端清空失败，仅清空本地显示')
  }
  messages.value = []
}

// ----- Scroll -----
function scrollToBottom() {
  nextTick(() => {
    if (chatBodyRef.value) {
      chatBodyRef.value.scrollTop = chatBodyRef.value.scrollHeight
    }
  })
}

// ----- Send -----
function onInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    onSend()
  }
}

async function onSend() {
  const text = inputText.value.trim()
  if (!text || isSending.value) return

  inputText.value = ''
  isSending.value = true

  // Add user message to history
  const userMsg = {
    _id: 'u-' + Date.now(),
    role: 'user',
    text,
    ts: Math.floor(Date.now() / 1000)
  }
  messages.value.push(userMsg)

  // Create active bot message
  const clientId = 'bot-' + Math.random().toString(36).slice(2)
  activeMsg.value = {
    clientId,
    hasContent: false,
    answerText: '',
    status: 'pending',
    doneList: [],
    runningList: [],
    imageUrls: [],
    error: '',
    time: nowTime()
  }

  scrollToBottom()

  try {
    if (isStream.value) {
      await handleStreamQuery(text)
    } else {
      await handleNonStreamQuery(text)
    }
  } catch (e) {
    if (activeMsg.value) {
      activeMsg.value.status = 'failed'
      activeMsg.value.error = e.message || '请求失败'
      activeMsg.value.hasContent = true
    }
  }
}

async function handleNonStreamQuery(text) {
  const data = await submitQuery(text, sessionId, false)

  if (activeMsg.value) {
    activeMsg.value.status = 'completed'
    activeMsg.value.answerText = data.answer || ''
    activeMsg.value.doneList = data.done_list || []
    activeMsg.value.imageUrls = data.image_urls || []
    activeMsg.value.hasContent = true
  }

  finalizeActiveMsg()
  isSending.value = false
}

async function handleStreamQuery(text) {
  const data = await submitQuery(text, sessionId, true)
  // data: { message, session_id }

  // Create SSE connection
  const es = createStreamSource(sessionId)
  let rawAnswerText = ''

  es.addEventListener('progress', (e) => {
    try {
      const d = JSON.parse(e.data || '{}')
      if (activeMsg.value) {
        activeMsg.value.status = d.status || 'processing'
        activeMsg.value.doneList = d.done_list || []
        activeMsg.value.runningList = d.running_list || []

        if (d.status === 'completed') {
          activeMsg.value.hasContent = true
        }
      }
    } catch { /* ignore parse errors */ }
  })

  es.addEventListener('delta', (e) => {
    try {
      const d = JSON.parse(e.data || '{}')
      const delta = d.delta || ''
      if (delta && activeMsg.value) {
        rawAnswerText += delta
        activeMsg.value.answerText = rawAnswerText
        activeMsg.value.hasContent = true
        scrollToBottom()
      }
    } catch { /* ignore */ }
  })

  es.addEventListener('final', (e) => {
    if (activeMsg.value) {
      activeMsg.value.status = 'completed'
      activeMsg.value.hasContent = true
      try {
        const d = JSON.parse(e.data || '{}')
        if (d.answer && d.answer.trim()) {
          activeMsg.value.answerText = d.answer
        } else {
          activeMsg.value.answerText = rawAnswerText || activeMsg.value.answerText
        }
        activeMsg.value.imageUrls = d.image_urls || []
      } catch {
        activeMsg.value.answerText = rawAnswerText || activeMsg.value.answerText
      }
    }
    es.close()
    finalizeActiveMsg()
    isSending.value = false
  })

  es.addEventListener('final_answer', (e) => {
    if (activeMsg.value) {
      activeMsg.value.status = 'completed'
      activeMsg.value.hasContent = true
      try {
        const d = JSON.parse(e.data || '{}')
        if (d.answer && d.answer.trim()) {
          activeMsg.value.answerText = d.answer
        }
        activeMsg.value.imageUrls = d.image_urls || []
      } catch { /* ignore */ }
    }
    es.close()
    finalizeActiveMsg()
    isSending.value = false
  })

  es.addEventListener('error', (e) => {
    if (activeMsg.value) {
      activeMsg.value.status = 'failed'
      activeMsg.value.hasContent = true
      try {
        const msg = e?.data ? (JSON.parse(e.data).error || 'SSE 连接中断') : 'SSE 连接中断'
        activeMsg.value.error = msg
      } catch {
        activeMsg.value.error = 'SSE 连接中断/失败'
      }
    }
    es.close()
    finalizeActiveMsg()
    isSending.value = false
  })

  // Fallback close on ready (edge case)
  es.addEventListener('ready', () => {
    // Connection established
  })
}

function finalizeActiveMsg() {
  if (!activeMsg.value) return

  const msg = activeMsg.value
  // Push to history
  messages.value.push({
    _id: 'b-' + Date.now(),
    role: 'bot',
    text: msg.answerText || '',
    image_urls: msg.imageUrls || [],
    ts: Math.floor(Date.now() / 1000)
  })

  activeMsg.value = null
  scrollToBottom()
}

// Auto-resize textarea
watch(inputText, () => {
  nextTick(() => {
    const el = inputRef.value
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 140) + 'px'
    }
  })
})
</script>

<style scoped>
.query-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Chat Body */
.chat-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 24px 10px;
  background: linear-gradient(180deg, rgba(26, 167, 255, 0.04), transparent 40%),
              linear-gradient(0deg, rgba(91, 124, 250, 0.04), transparent 45%);
}

.dayline {
  text-align: center;
  color: var(--muted);
  font-size: 12px;
  margin: 4px 0 14px;
}

/* Messages */
.msg {
  display: flex;
  gap: 10px;
  margin: 10px 0;
  align-items: flex-start;
}

.msg.user {
  justify-content: flex-end;
}

.msg-content {
  flex: 1;
  min-width: 0;
}

.msg.user .msg-content {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
}

.avatar {
  width: 38px;
  height: 38px;
  border-radius: 10px;
  background: #e9eef7;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: #3b4a5a;
  flex-shrink: 0;
  line-height: 1.2;
  text-align: center;
}

.bot-avatar {
  background: none;
}

.user-avatar {
  background: linear-gradient(135deg, var(--brand), var(--brand2));
}

.bubble {
  max-width: min(720px, 76%);
  display: inline-block;
  border-radius: 14px;
  padding: 10px 14px;
  border: 1px solid var(--border);
  background: var(--bubble-bot);
  line-height: 1.55;
  font-size: 14px;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

.msg.user .bubble {
  background: var(--bubble-user);
  color: #fff;
  border-color: transparent;
}

.meta {
  font-size: 11px;
  color: var(--muted);
  margin-top: 5px;
}

.msg.user .meta {
  color: rgba(255, 255, 255, 0.8);
}

/* Typing */
.typing {
  display: inline-flex;
  gap: 6px;
  align-items: center;
  padding: 4px 0;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #9aa6b2;
  animation: bounce 1.2s infinite ease-in-out;
}

.dot:nth-child(2) { animation-delay: .15s; }
.dot:nth-child(3) { animation-delay: .3s; }

/* Progress Details */
.progress-details {
  margin-top: 8px;
  border: 1px dashed rgba(122, 134, 154, 0.45);
  border-radius: 12px;
  padding: 8px 10px;
  background: rgba(122, 134, 154, 0.05);
}

.progress-details summary {
  cursor: pointer;
  user-select: none;
  color: var(--muted);
  font-size: 12px;
}

.progress-list {
  margin: 8px 0 0 18px;
  padding: 0;
  list-style: none;
}

.progress-list li {
  margin: 4px 0;
  font-size: 12px;
  color: #56657a;
}

.log-check {
  color: var(--brand);
  font-weight: 700;
  font-size: 14px;
}

.log-cross {
  color: var(--danger);
  font-weight: 700;
  font-size: 14px;
}

.error-msg {
  margin-top: 8px;
  color: var(--danger);
  font-size: 13px;
}

/* Composer */
.composer {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding: 16px 120px;
}

.composer textarea {
  flex: 1;
  resize: none;
  min-height: 44px;
  max-height: 140px;
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid var(--border);
  outline: none;
  font-size: 14px;
  line-height: 1.5;
  font-family: inherit;
  background: var(--panel);
  box-shadow: 0 2px 12px rgba(31, 45, 61, 0.08), 0 1px 3px rgba(31, 45, 61, 0.06);
  transition: box-shadow 0.2s;
}

.composer textarea:focus {
  border-color: rgba(26, 167, 255, 0.5);
  box-shadow: 0 4px 20px rgba(26, 167, 255, 0.15), 0 1px 3px rgba(31, 45, 61, 0.08);
}

.send-btn {
  width: 80px;
  height: 44px;
  flex-shrink: 0;
  box-shadow: 0 2px 12px rgba(26, 167, 255, 0.25), 0 1px 3px rgba(31, 45, 61, 0.1);
}

.send-btn:hover:not(:disabled) {
  box-shadow: 0 4px 20px rgba(26, 167, 255, 0.4);
}

/* Chat toolbar — fixed at top */
.chat-toolbar {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 12px;
  padding: 10px 24px 8px;
  flex-shrink: 0;
}

.toggle-label {
  font-size: 12px;
  color: var(--muted);
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: pointer;
  user-select: none;
}

.btn-clear {
  border: none;
  background: rgba(231, 76, 60, 0.15);
  color: #fff;
  padding: 5px 14px;
  border-radius: 8px;
  font-size: 12px;
  cursor: pointer;
  font-family: inherit;
  font-weight: 500;
  transition: background 0.2s;
}

.btn-clear:hover {
  background: rgba(231, 76, 60, 0.3);
}
</style>
