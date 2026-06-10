<template>
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" class="bot-avatar-svg">
    <!-- 六边形外框 -->
    <polygon
      points="24,4 42,14 42,34 24,44 6,34 6,14"
      fill="url(#bgGrad)"
      stroke="url(#strokeGrad)"
      stroke-width="1.5"
      stroke-linejoin="round"
    />
    <!-- 中心节点 -->
    <circle cx="24" cy="24" r="5" fill="white" opacity="0.95" />
    <circle cx="24" cy="24" r="2.5" fill="url(#nodeGrad)" />
    <!-- 连接线：上 -->
    <line x1="24" y1="14" x2="24" y2="19" stroke="white" stroke-width="1.2" opacity="0.7" />
    <!-- 连接线：下 -->
    <line x1="24" y1="29" x2="24" y2="34" stroke="white" stroke-width="1.2" opacity="0.7" />
    <!-- 连接线：左上 -->
    <line x1="15" y1="18" x2="19" y2="21" stroke="white" stroke-width="1.2" opacity="0.7" />
    <!-- 连接线：右上 -->
    <line x1="33" y1="18" x2="29" y2="21" stroke="white" stroke-width="1.2" opacity="0.7" />
    <!-- 连接线：左下 -->
    <line x1="15" y1="30" x2="19" y2="27" stroke="white" stroke-width="1.2" opacity="0.7" />
    <!-- 连接线：右下 -->
    <line x1="33" y1="30" x2="29" y2="27" stroke="white" stroke-width="1.2" opacity="0.7" />
    <!-- 卫星小节点 -->
    <circle cx="24" cy="14.5" r="1.8" fill="white" opacity="0.85" />
    <circle cx="24" cy="33.5" r="1.8" fill="white" opacity="0.85" />
    <circle cx="15.5" cy="18.5" r="1.8" fill="white" opacity="0.85" />
    <circle cx="32.5" cy="18.5" r="1.8" fill="white" opacity="0.85" />
    <circle cx="15.5" cy="29.5" r="1.8" fill="white" opacity="0.85" />
    <circle cx="32.5" cy="29.5" r="1.8" fill="white" opacity="0.85" />

    <defs>
      <linearGradient id="bgGrad" x1="6" y1="4" x2="42" y2="44">
        <stop offset="0%" stop-color="#1aa7ff" stop-opacity="0.35" />
        <stop offset="100%" stop-color="#5b7cfa" stop-opacity="0.35" />
      </linearGradient>
      <linearGradient id="strokeGrad" x1="6" y1="4" x2="42" y2="44">
        <stop offset="0%" stop-color="#1aa7ff" stop-opacity="0.6" />
        <stop offset="100%" stop-color="#5b7cfa" stop-opacity="0.6" />
      </linearGradient>
      <linearGradient id="nodeGrad" x1="21" y1="21" x2="27" y2="27">
        <stop offset="0%" stop-color="#1aa7ff" />
        <stop offset="100%" stop-color="#5b7cfa" />
      </linearGradient>
    </defs>
  </svg>
</template>

<style scoped>
.bot-avatar-svg {
  width: 38px;
  height: 38px;
  display: block;
}
</style>
