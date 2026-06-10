<template>
  <aside class="sidebar">
    <div class="sidebar-brand">
      <BrainLogo />
      <div class="sidebar-title">
        <strong>基石智库</strong>
        <span>知识库管理平台</span>
      </div>
    </div>

    <nav class="sidebar-nav">
      <router-link to="/import" class="nav-item" :class="{ active: isActive('/import') }">
        <span class="nav-icon">📄</span>
        <span>文档上传</span>
      </router-link>
      <router-link to="/query" class="nav-item" :class="{ active: isActive('/query') }">
        <span class="nav-icon">💬</span>
        <span>知识库查询</span>
      </router-link>
    </nav>

    <div class="sidebar-footer">
      <div class="footer-status">
        <div class="status-item">
          <span class="status-dot" :class="{ offline: !importOnline }"></span>
          <span>文档上传</span>
        </div>
        <div class="status-item">
          <span class="status-dot" :class="{ offline: !queryOnline }"></span>
          <span>查询搜索</span>
        </div>
      </div>
      <div class="footer-divider"></div>
      <div class="footer-tagline">系统运行中</div>
    </div>
  </aside>
</template>

<script setup>
import { useRoute } from 'vue-router'
import BrainLogo from './BrainLogo.vue'

defineProps({
  importOnline: { type: Boolean, default: false },
  queryOnline: { type: Boolean, default: false }
})

const route = useRoute()

function isActive(path) {
  return route.path === path
}
</script>
