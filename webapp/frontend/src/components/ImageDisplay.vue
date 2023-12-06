<template>
  <div>
    <v-container class="fixed-center">
      <v-row>
        <v-col class="d-flex justify-center">
          <img v-if="resizedImage" :src="resizedImage" alt="Generated Image" />
          <div v-else class="grey-placeholder"></div>
        </v-col>
      </v-row>
    </v-container>
  </div>
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
  z-index: 2;
}
.fixed-center {
  position: fixed; /* Fix position relative to the viewport */
  top: 50%; /* Position at 50% from the top */
  left: 50%; /* Position at 50% from the left */
  transform: translate(-50%, -50%); /* Adjust the positioning to center the element */
}
.img {
  z-index: 2;
}


</style>
