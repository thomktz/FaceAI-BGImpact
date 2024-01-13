import { createRouter, createWebHistory } from "vue-router";
import Home from "@/views/Home.vue";
import MainLayout from "@/views/MainLayout.vue";
import GANSpace from "@/views/GANSpace.vue";
import VAESpace from "@/views/VAESpace.vue";

const routes = [
  {
    path: "/",
    component: MainLayout,
    children: [
      {
        path: "",
        name: "Home",
        component: Home,
      },
      {
        path: "/stylegan",
        name: "StyleGAN",
        component: GANSpace,
      },
      {
        path: "/vae",
        name: "VAE",
        component: VAESpace,
      },
    ],
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
