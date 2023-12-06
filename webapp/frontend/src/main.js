import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

// Vuetify
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'
import VueKatex from '@hsorby/vue3-katex';
import 'katex/dist/katex.min.css';

const vuetify = createVuetify({
  components,
  directives,
})

createApp(App).use(router).use(vuetify).use(VueKatex).mount('#app')