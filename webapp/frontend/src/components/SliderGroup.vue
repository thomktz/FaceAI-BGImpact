<template>
  <div>
    <v-slider v-for="index in nSliders" :key="index" v-model="sliderValues[index]" @change="logSliderValue(index)" thumb-label="always"></v-slider>
    <!-- Buttons -->
    <v-btn @click="setSlidersToZero">Set Sliders to 0</v-btn>
    <v-btn @click="setSlidersToRandom">Set Sliders to Random</v-btn>
  </div>
</template>

<script>
export default {
  props: {
    nSliders: {
      type: Number,
      default: 10
    }
  },
  data: () => ({
    sliderValues: []
  }),
  created() {
    this.sliderValues = Array(this.nSliders).fill(0);
  },
  methods: {
    logSliderValue(index) {
      console.log(`Slider ${index}: ${this.sliderValues[index]}`);
    },
    setSlidersToZero() {
      this.sliderValues.fill(0);
    },
    setSlidersToRandom() {
      this.sliderValues = this.sliderValues.map(() => Math.random());
    }
  }
}
</script>
