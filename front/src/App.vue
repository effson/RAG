<template>
  <div class="layout">
    <Sidebar :import-online="importOnline" :query-online="queryOnline" />
    <div class="main">
      <div class="page-content">
        <router-view />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import Sidebar from './components/Sidebar.vue'
import { healthCheck } from './api/query'

const queryOnline = ref(false)
const importOnline = ref(false)

let healthTimer = null

async function checkHealth() {
  try {
    const h = await healthCheck()
    queryOnline.value = h.ok === true
  } catch {
    queryOnline.value = false
  }

  try {
    // FastAPI 自动提供 /openapi.json，用这个探测导入服务是否可达
    const res = await fetch('/import-api/openapi.json')
    importOnline.value = res.ok
  } catch {
    // 网络错误（连接拒绝等）→ 服务离线
    importOnline.value = false
  }
}

onMounted(() => {
  checkHealth()
  healthTimer = setInterval(checkHealth, 15000)
})

onUnmounted(() => {
  if (healthTimer) clearInterval(healthTimer)
})
</script>
