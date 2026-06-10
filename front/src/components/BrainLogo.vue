<template>
  <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" class="brain-logo-svg">
    <!-- 大脑皮层外轮廓（侧面） -->
    <path
      d="M32 4C20 4 14 6 8 14C4 19 2 24 2 28C2 32 5 36 9 38C11 39 14 39 16 38C18 37 19 35 19 34V32C18 30 16 28 15 26C13 23 13 19 15 16C17 13 21 10 27 9C32 8 37 9 41 13C44 17 46 22 46 27C46 32 44 36 40 39C37 41 33 42 29 42C25 42 22 40 20 38C18 36 18 34 18 32V28"
      fill="none"
      stroke="url(#brainSideGrad)"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
    <!-- 小脑（后下方） -->
    <path
      d="M18 28C16 31 14 34 13 37C12 39 14 41 17 40C19 39 20 37 20 35V32"
      fill="none"
      stroke="url(#brainSideGrad2)"
      stroke-width="1.8"
      stroke-linecap="round"
      stroke-linejoin="round"
    />
    <!-- 脑干 -->
    <line x1="17" y1="38" x2="17" y2="44" stroke="white" stroke-width="1.5" opacity="0.5" stroke-linecap="round" />
    <line x1="22" y1="36" x2="22" y2="44" stroke="white" stroke-width="1.2" opacity="0.35" stroke-linecap="round" />
    <!-- 内部沟回纹路 -->
    <path d="M12 20 Q16 24 14 28" fill="none" stroke="white" stroke-width="1" opacity="0.45" stroke-linecap="round" />
    <path d="M22 12 Q26 18 24 24" fill="none" stroke="white" stroke-width="1" opacity="0.4" stroke-linecap="round" />
    <path d="M32 13 Q36 20 34 28" fill="none" stroke="white" stroke-width="1" opacity="0.45" stroke-linecap="round" />
    <path d="M40 18 Q42 24 40 30" fill="none" stroke="white" stroke-width="1" opacity="0.4" stroke-linecap="round" />
    <!-- 神经网络节点 -->
    <circle cx="14" cy="24" r="1.5" fill="white" opacity="0.6" />
    <circle cx="26" cy="16" r="1.5" fill="white" opacity="0.6" />
    <circle cx="36" cy="22" r="1.5" fill="white" opacity="0.6" />
    <circle cx="30" cy="34" r="1.5" fill="white" opacity="0.5" />
    <!-- 数据脉冲（额叶区光点） -->
    <circle cx="42" cy="20" r="2" fill="white" opacity="0.7" />
    <circle cx="42" cy="20" r="3.5" fill="white" opacity="0.15" />

    <defs>
      <linearGradient id="brainSideGrad" x1="2" y1="4" x2="46" y2="42">
        <stop offset="0%" stop-color="#fff" stop-opacity="0.95" />
        <stop offset="40%" stop-color="#1aa7ff" stop-opacity="0.6" />
        <stop offset="100%" stop-color="#5b7cfa" stop-opacity="0.35" />
      </linearGradient>
      <linearGradient id="brainSideGrad2" x1="13" y1="37" x2="20" y2="35">
        <stop offset="0%" stop-color="#5b7cfa" stop-opacity="0.6" />
        <stop offset="100%" stop-color="#fff" stop-opacity="0.5" />
      </linearGradient>
    </defs>
  </svg>
</template>

<style scoped>
.brain-logo-svg {
  width: 32px;
  height: 32px;
  display: block;
}
</style>
