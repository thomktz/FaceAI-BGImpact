import { createRouter, createWebHistory } from 'vue-router'
import Home from '@/views/Home.vue'
import MainLayout from '@/views/MainLayout.vue';
import LiveDemo from '@/views/LiveDemo.vue';
import GANSpace from '@/views/GANSpace.vue';

const routes = [
  {
    path: '/',
    component: MainLayout,
    children: [
      {
        path: '',
        name: 'Home',
        component: Home
      },
      {
        path: '/live-demo',
        name: 'LiveDemo',
        component: LiveDemo
      },
      {
        path: '/gan-space',
        name: 'GANSpace',
        component: GANSpace
      },
    ]
  },
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
