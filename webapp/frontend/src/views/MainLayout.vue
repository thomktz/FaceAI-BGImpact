<template>
  <header class="header flex items-center justify-between p-4">
    <h1 class="text-xl font-bold ml-8">{{ headerTitle }}</h1>
    <nav class="right-section">
      <!-- Navigation links here -->
      <router-link to="/" class="hover:text-gray-100 px-2">Home</router-link>
      <router-link to="/live-demo" class="hover:text-gray-100 px-2"
        >Live Demo</router-link
      >
      <router-link to="/gan-space" class="hover:text-gray-100 px-2"
        >GAN Space</router-link
      >
      <!-- Add more tabs here -->
    </nav>
  </header>
  <main class="fill-height">
    <router-view @updateTitle="setTitle" />
  </main>
</template>

<script>
import { ref } from "vue";

export default {
  name: "MainLayout",
  setup() {
    const headerTitle = ref("Advanced Machine Learning Project");
    const setTitle = (newTitle) => {
      headerTitle.value = newTitle;
    };

    return { headerTitle, setTitle };
  },
};
</script>

<style>
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  background-color: #4c7abb;
  box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
}

.right-section {
  display: flex;
  gap: 10px;
}

h1 {
  font-size: 2rem;
  font-weight: 700;
}
</style>
