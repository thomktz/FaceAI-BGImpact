<template>
  <v-container>
    <v-row>
      <v-col class="d-flex justify-center">
        <img v-if="resizedImage" :src="resizedImage" alt="Generated Image" />
        <div v-else class="grey-placeholder"></div>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  props: {
    imageData: String
  },
  methods: {
    resizeImage(imageData) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.width = 400;
          canvas.height = 400;
          ctx.drawImage(img, 0, 0, 400, 400);
          const resizedImageData = canvas.toDataURL('image/jpeg');
          resolve(resizedImageData);
        };
        img.onerror = reject;
        img.src = imageData;
      });
    }
  },
  watch: {
    imageData(newImageData) {
      if (newImageData) {
        this.resizeImage(newImageData).then(resizedImageData => {
          this.resizedImage = resizedImageData;
        }).catch(error => {
          console.error('Error resizing image:', error);
        });
      }
    }
  },
  data() {
    return {
      resizedImage: ''
    };
  }
}
</script>

<style>
.grey-placeholder {
  width: 400px;
  height: 400px;
  background-color: grey;
}
</style>
