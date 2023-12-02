<template>
  <v-container class="fill-height" fluid>
    <v-row align="center" justify="center">
      <!-- Image Grid Display -->
      <v-col cols="6">
        <v-row>
          <v-col cols="4" v-for="(image, index) in images" :key="index">
            <img :src="image" alt="Generated Image" class="resized-image" />
          </v-col>
        </v-row>
      </v-col>

      <!-- Control Panel -->
      <v-col cols="6">
        <div class="control-panel">
          <!-- Sliders for Style Vectors -->
          <v-row v-for="(value, index) in styleVector" :key="index">
            <v-col cols="12">
              <v-slider
                v-model="styleVector[index]"
                :min="-3"
                :max="3"
                :step="0.01"
                thumb-label="always"
                @end="applyStyleSliders"
              ></v-slider>
            </v-col>
          </v-row>

          <!-- Buttons -->
          <v-row>
            <v-col cols="12">
              <v-btn block color="primary" class="generate-button" @click="randomizeLatents">
                Randomize Latents
              </v-btn>
              <v-btn block color="primary" class="generate-button" @click="applyStyleSliders">
                Apply Style Sliders
              </v-btn>
              <v-btn block color="primary" class="generate-button" @click="randomizeStyleVectors">
                Randomize Style Vectors
              </v-btn>
              <v-btn block color="primary" class="generate-button" @click="resetStyleSliders">
                Reset
              </v-btn>
            </v-col>
          </v-row>
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
      images: [],  // Store images URLs
      styleVector: new Array(10).fill(0),  // Initialize style vector
    };
  },
  methods: {
    randomizeLatents() {
      apiClient.get('stylegan/randomize-latents')
        .then(response => {
          console.log("Generation took:", response.data.time)
          this.images = response.data.images.map(img => `data:image/jpeg;base64,${img}`);
          console.log("Generation took:", response.data.time)
        })
        .catch(error => console.error('Error randomizing latents:', error));
    },
    applyStyleSliders() {
      const payload = { slider_values: this.styleVector };
      apiClient.post('stylegan/apply-style-sliders', payload)
        .then(response => {
          this.images = response.data.images.map(img => `data:image/jpeg;base64,${img}`);
          console.log("Generation took:", response.data.time)
        })
        .catch(error => console.error('Error applying style sliders:', error));
    },
    randomizeStyleVectors() {
      this.styleVector = this.styleVector.map(() => Math.random() * 2 - 1);
      this.applyStyleSliders();
    },
    resetStyleSliders() {
      this.styleVector.fill(0);
      this.applyStyleSliders();
    }
  }
};
</script>

<style scoped>
.resized-image {
  width: 100%;
  height: auto;
  object-fit: contain;
}
.control-panel {
  padding: 10px;
  display: flex;
  flex-direction: column;
  align-items: stretch;  /* Updated to stretch for consistent width */
}
.generate-button {
  margin-top: 10px;
}
</style>
