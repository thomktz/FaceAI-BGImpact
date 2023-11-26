<template>
  <v-container class="fill-height" fluid>
    <v-row align="center" justify="center">
      <v-col cols="6">
        <!-- Placeholder grey div if no image is available -->
        <div v-if="!imageUrl" class="placeholder-container"></div>
        <!-- Generated image if available -->
        <img v-else :src="imageUrl" alt="Random StyleGAN Image" class="resized-image" />
      </v-col>
      <v-col cols="3">
        <!-- Control Panel -->
        <div class="control-panel">
          <!-- Sliders for controlling the latent vector -->
          <v-slider
            v-for="(value, index) in latentVector"
            :key="index"
            v-model="latentVector[index]"
            :min="-1"
            :max="1"
            :step="0.01"
            @end="updateImageFromLatent"
            thumb-label="always"
          ></v-slider>
          <!-- Button to generate a random latent vector -->
          <v-btn color="primary" class="generate-button" @click="randomizeLatentVector">
            Randomize Latent Vector
          </v-btn>
          <v-btn color="primary" class="generate-button" @click="updateImageFromLatent">
            Get current vector
          </v-btn>
        </div>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import { apiClient } from '@/apiConfig'

export default {
  data() {
    return {
      imageUrl: null,
      latentVector: new Array(10).fill(0), // Initialize the latent vector with zeros
    };
  },
  methods: {
    randomizeLatentVector() {
      this.latentVector = this.latentVector.map(() => (Math.random() * 2) - 1);
      this.updateImageFromLatent();
    },
    updateImageFromLatent() {
      console.log('Updating image from latent vector:', this.latentVector);
      // Use the current state of latentVector to generate an image
      const payload = {
        latent_vector: this.latentVector,
      };
      apiClient.post('/stylegan/from-latent', payload)
        .then(response => {
          const base64Image = response.data.image;
          this.imageUrl = `data:image/jpeg;base64,${base64Image}`;
        })
        .catch(error => {
          console.error('Error generating image from latent space:', error);
        });
    }
  }
};
</script>

<style scoped>
/* Existing styles remain unchanged */
.placeholder-container {
  width: 512px;
  height: 512px;
  background-color: #e0e0e0;
}
.resized-image {
  width: 512px;
  height: 512px;
  object-fit: contain;
}
.control-panel {
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  height: 100%;
}
.generate-button {
  margin-top: auto;
}
</style>
