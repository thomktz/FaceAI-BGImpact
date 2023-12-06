import { createApp } from "vue";
import App from "./App.vue";
import router from "./router";

// Vuetify
import vuetify from "./plugins/vuetify";
import "vuetify/dist/vuetify.min.css";

import VueKatex from "@hsorby/vue3-katex";
import "katex/dist/katex.min.css";

createApp(App).use(router).use(vuetify).use(VueKatex).mount("#app");
