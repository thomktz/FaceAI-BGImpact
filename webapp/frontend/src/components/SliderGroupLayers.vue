<template>
  <div>
    <div v-for="index in nSliders" :key="index">
      <v-slider 
        v-model="sliderValues[index-1]" 
        :min="-slidersRange" 
        :max="slidersRange" 
        :step="0.01 * slidersRange * 5"
        thumb-label="always">
      </v-slider>
      <v-range-slider 
        v-model="layerRanges[index-1]" 
        :ticks="[0, 1, 2, 3, 4]"
        :min="0"
        :max="4" 
        :step="1" 
        show-ticks="always"
        :tick-size="4">
      </v-range-slider>
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
      default: 4
    },
    slidersRange: {
      type: Number,
      default: 1.0
    }
  },
  data() {
    return {
      sliderValues: Array(this.nSliders).fill(0),
      layerRanges: Array(this.nSliders).fill().map(() => [0, this.maxLayers])
    };
  },
  watch: {
    sliderValues: {
      handler(newValues) {
        this.$emit('newSliderChange', newValues);
      },
      deep: true
    },
    layerRanges: {
      handler(newRanges) {
        const fullLayerLists = newRanges.map(range => {
          const [lower, upper] = range;
          return Array.from({ length: upper - lower + 1 }, (_, i) => i + lower);
        });
        this.$emit('layerListChange', fullLayerLists);
      },
      deep: true
    }
  },
  methods: {
    setSlidersToZero() {
      this.sliderValues.fill(0);
      this.layerRanges = this.layerRanges.map(() => [0, this.maxLayers]);
    }
    // No need for logSliderValue method since watchers are handling logging
  }
}
</script>
