<template>
  <div class="import-page">
    <h2 class="section-heading">📄 知识库文件导入</h2>
    <p class="section-desc">支持 PDF 和 Markdown 文件，上传后自动解析、切分、向量化并导入知识库</p>

      <!-- Drop Zone -->
      <div
        class="drop-zone"
        :class="{ 'drop-zone-active': isDragging }"
        @click="triggerFileInput"
        @dragover.prevent="onDragOver"
        @dragleave.prevent="onDragLeave"
        @drop.prevent="onDrop"
      >
        <div class="drop-icon">☁️</div>
        <p class="drop-text">点击或拖拽文件到此处</p>
        <p class="drop-hint">支持 .pdf / .md 格式，可多选</p>
        <input
          ref="fileInputRef"
          type="file"
          multiple
          accept=".pdf,.md"
          class="file-input-hidden"
          @change="onFileInputChange"
        />
      </div>

      <!-- File List -->
      <div v-if="files.length > 0" class="file-list">
        <div
          v-for="f in files"
          :key="f.id"
          class="file-item fade-up"
        >
          <div class="file-main">
            <div class="file-icon">{{ fileIcon(f.name) }}</div>
            <div class="file-info">
              <div class="file-name">{{ f.name }}</div>
              <div class="file-meta">
                <span>{{ f.sizeFormatted }}</span>
                <span v-if="f.taskId" class="file-task-id">ID: {{ f.taskId.slice(0, 8) }}…</span>
              </div>
              <!-- Progress bar -->
              <div v-if="f.status === 'uploading' || f.status === 'processing'" class="progress-bar-wrap">
                <div
                  class="progress-bar-fill"
                  :class="{ 'fill-processing': f.status === 'processing' }"
                  :style="{ width: f.progress + '%' }"
                ></div>
              </div>
              <!-- Log -->
              <details v-if="f.status !== 'idle'" class="log-details" :open="f.status === 'processing'">
                <summary>
                  日志（已完成 {{ f.doneList.length }}，进行中 {{ f.runningList.length }}）
                </summary>
                <ul class="log-list">
                  <li v-if="f.doneList.length === 0 && f.runningList.length === 0">暂无日志</li>
                  <li v-for="item in f.doneList" :key="'d-' + item">
                    <span class="log-check">✓</span> {{ item }}已完成
                  </li>
                  <li v-for="item in f.runningList" :key="'r-' + item">
                    <span v-if="f.status === 'error'" class="log-cross">✗</span>
                    <span v-else class="spinner"></span>
                    正在进行{{ item }}...
                  </li>
                </ul>
              </details>
            </div>
          </div>
          <span class="badge" :class="'badge-' + f.status">
            {{ statusLabel(f.status) }}
          </span>
        </div>
      </div>

      <!-- Empty State -->
      <div v-else class="empty-state">
        <div class="empty-icon">📂</div>
        <p>暂无上传文件</p>
      </div>
    </div>
</template>

<script setup>
import { ref, onUnmounted } from 'vue'
import { useImportStore } from '../stores/importStore'

const { files, startUpload, cleanupCompletedTimers } = useImportStore()

const fileInputRef = ref(null)
const isDragging = ref(false)

function triggerFileInput() {
  fileInputRef.value?.click()
}

function onDragOver() {
  isDragging.value = true
}

function onDragLeave() {
  isDragging.value = false
}

function onDrop(e) {
  isDragging.value = false
  handleFiles(e.dataTransfer.files)
}

function onFileInputChange(e) {
  handleFiles(e.target.files)
  e.target.value = ''
}

function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase()
  return ext === 'pdf' ? '📕' : '📝'
}

function statusLabel(s) {
  const map = {
    idle: '等待中',
    uploading: '上传中…',
    processing: '处理中…',
    completed: '✓',
    error: '✗'
  }
  return map[s] || s
}

function handleFiles(fileList) {
  for (const file of fileList) {
    startUpload(file)
  }
}

onUnmounted(() => {
  // 组件卸载时只清理已完成的定时器，进行中的继续跑
  cleanupCompletedTimers()
})
</script>

<style scoped>
.import-page {
  height: 100%;
  overflow-y: auto;
  padding: 24px 36px;
}

.section-heading {
  font-size: 20px;
  margin-bottom: 4px;
  color: var(--text);
}

.section-desc {
  color: var(--muted);
  font-size: 13px;
  margin-bottom: 24px;
}

/* Drop Zone */
.drop-zone {
  border: 2px dashed #b0c4de;
  border-radius: 12px;
  padding: 48px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.25s;
  background: var(--panel);
}

.drop-zone:hover {
  border-color: var(--brand);
  background: #f0f7ff;
}

.drop-zone-active {
  border-color: var(--brand);
  background: #e8f4fd;
}

.drop-icon {
  font-size: 44px;
  margin-bottom: 10px;
}

.drop-text {
  font-size: 15px;
  color: var(--text);
  margin-bottom: 6px;
}

.drop-hint {
  font-size: 12px;
  color: var(--muted);
}

.file-input-hidden {
  display: none;
}

/* File List */
.file-list {
  margin-top: 24px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.file-item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: #fff;
  transition: box-shadow 0.2s;
}

.file-item:hover {
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.file-main {
  display: flex;
  gap: 12px;
  flex: 1;
  min-width: 0;
}

.file-icon {
  font-size: 28px;
  flex-shrink: 0;
  margin-top: 2px;
}

.file-info {
  flex: 1;
  min-width: 0;
}

.file-name {
  font-weight: 600;
  font-size: 14px;
  color: var(--text);
  word-break: break-all;
}

.file-meta {
  font-size: 12px;
  color: var(--muted);
  margin-top: 2px;
  display: flex;
  gap: 12px;
}

.file-task-id {
  font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 11px;
}

/* Progress */
.progress-bar-wrap {
  width: 100%;
  height: 4px;
  background: #eef1f5;
  border-radius: 2px;
  margin-top: 8px;
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  background: var(--warning);
  border-radius: 2px;
  transition: width 0.4s, background 0.4s;
}

.progress-bar-fill.fill-processing {
  background: var(--brand);
}

/* Log */
.log-details {
  margin-top: 8px;
  font-size: 12px;
  color: var(--muted);
}

.log-details summary {
  cursor: pointer;
  color: #6c7a89;
  user-select: none;
}

.log-list {
  margin: 6px 0 0 16px;
  padding: 0;
  list-style: none;
}

.log-list li {
  margin: 3px 0;
  line-height: 1.5;
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


/* Empty */
.empty-state {
  margin-top: 32px;
  text-align: center;
  color: var(--muted);
}

.empty-icon {
  font-size: 48px;
  margin-bottom: 10px;
}
</style>
