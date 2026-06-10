import { createRouter, createWebHashHistory } from 'vue-router'
import ImportView from '../views/ImportView.vue'
import QueryView from '../views/QueryView.vue'

const routes = [
  {
    path: '/',
    redirect: '/query'
  },
  {
    path: '/import',
    name: 'import',
    component: ImportView,
    meta: { title: '文档上传' }
  },
  {
    path: '/query',
    name: 'query',
    component: QueryView,
    meta: { title: '知识库查询' }
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router
