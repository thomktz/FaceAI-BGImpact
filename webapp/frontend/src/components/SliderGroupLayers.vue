<template>
  <div>
    <div v-for="index in nSliders" :key="index">
      <v-slider v-model="sliderValues[index]" @change="logSliderValue(index)" thumb-label="always"></v-slider>
      <v-range-slider v-model="layerRanges[index]" :max="maxLayers" thumb-label="always"></v-range-slider>
    </div>
    <!-- Buttons -->
    <v-btn @click="setSlidersToZero">Set Sliders to 0</v-btn>
  </div>
</template>

<script>
export default {
  props: {
    nSliders: {
      type: Number,
      default: 5
    },
    maxLayers: {
      type: Number,
      default: 10
    }
  },
  data: () => ({
    sliderValues: [],
    layerRanges: []
  }),
  created() {
    this.sliderValues = Array(this.nSliders).fill(0);
    this.layerRanges = Array(this.nSliders).fill().map(() => [0, this.maxLayers]);
  },
  methods: {
    logSliderValue(index) {
      console.log(`Slider ${index}: Value = ${this.sliderValues[index]}, Layers = ${this.layerRanges[index]}`);
    },
    setSlidersToZero() {
      this.sliderValues.fill(0);
    }
  }
}
</script>
