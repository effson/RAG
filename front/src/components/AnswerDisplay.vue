<template>
  <div class="answer-display">
    <div v-if="displayText" class="answer-text">{{ displayText }}</div>
    <div v-if="displayText && displayImages.length > 0" class="answer-gap"></div>
    <div v-if="displayImages.length > 0" class="answer-images">
      <div v-for="(url, i) in displayImages" :key="i" class="image-wrapper">
        <img
          :src="url"
          :alt="'参考图片 ' + (i + 1)"
          loading="lazy"
          referrerpolicy="no-referrer"
          @error="onImgError($event)"
        />
        <a :href="url" target="_blank" rel="noopener noreferrer" class="image-link">
          {{ truncateUrl(url) }}
        </a>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import {
  parseAnswerAndImages,
  collectImageUrls,
  normalizeUrl,
  extractUrls,
  isImageUrl
} from '../utils/helpers'

const props = defineProps({
  text: { type: String, default: '' },
  imageUrls: { type: Array, default: () => [] }
})

const displayText = computed(() => {
  const { text } = parseAnswerAndImages(props.text || '')
  return text
})

const displayImages = computed(() => {
  return collectImageUrls(props.text || '', props.imageUrls || [])
})

function truncateUrl(url) {
  const s = String(url || '')
  if (s.length <= 60) return s
  return s.slice(0, 30) + '…' + s.slice(-27)
}

function onImgError(e) {
  e.target.style.display = 'none'
  // Also hide the parent wrapper if available
  const wrapper = e.target.closest('.image-wrapper')
  if (wrapper) wrapper.style.display = 'none'
}
</script>

<style scoped>
.answer-display {
  /* container */
}

.answer-text {
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

.answer-gap {
  height: 10px;
}

.answer-images {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.image-wrapper {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.image-wrapper img {
  display: block;
  max-width: min(100%, 600px);
  max-height: 600px;
  width: auto;
  height: auto;
  border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: 0 10px 24px rgba(31, 45, 61, 0.1);
  background: #fff;
}

.image-link {
  font-size: 11px;
  color: var(--muted);
  text-decoration: none;
  word-break: break-all;
}

.image-link:hover {
  color: var(--brand);
  text-decoration: underline;
}
</style>
