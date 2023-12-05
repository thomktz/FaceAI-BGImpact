<template>
  <div>
    <v-slider 
      v-for="index in nSliders" 
      :key="index" 
      :min="-slidersRange" :max="slidersRange" 
      :step="0.01 * slidersRange * 5"
      v-model="sliderValues[index - 1]" 
      thumb-label="always">
    </v-slider>
    <!-- Buttons -->
    <v-btn @click="setSlidersToZero">Set Sliders to 0</v-btn>
    <v-btn @click="setSlidersToRandom">Set Sliders to Random</v-btn>
    <v-btn @click="setOthersToZero">Set Others to 0</v-btn>
    <v-btn @click="setOthersToRandom">Set Others to Random</v-btn>
  </div>
</template>

<script>
export default {
  props: {
    nSliders: {
      type: Number,
      default: 10
    },
    slidersRange: {
      type: Number,
      default: 1.0
    }
  },
  data() {
    return {
      sliderValues: Array(this.nSliders).fill(0)
    };
  },
  watch: {
    sliderValues: {
      handler(newValues) {
        this.$emit('originalSliderChange', newValues);
      },
      deep: true
    }
  },

  methods: {
    setSlidersToZero() {
      this.sliderValues.fill(0);
    },
    setSlidersToRandom() {
      this.sliderValues = this.sliderValues.map(() => Math.random() * (this.slidersRange * 2) - this.slidersRange);
    },
    setOthersToZero() {
      this.$emit('othersToZero');
    },
    setOthersToRandom() {
      this.$emit('othersToRandom');
    }
  }
}
</script>
