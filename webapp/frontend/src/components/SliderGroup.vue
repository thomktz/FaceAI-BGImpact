<template>
  <!-- Content of original-def with text and formulas goes here -->
  <div class="original-def">
    <div class="inline-container">
      The StyleGAN generator transforms a 256-dim "noise" vector into a 256-dim
      "style" vector
      <div v-katex="'\\mathbb{w} = \\mathcal{M}(z)'"></div>
      where
      <div v-katex="'\\mathcal{M}'"></div>
      is the StyleGAN Mapping network. This "style" is then used to generate an
      image
      <div v-katex="'img = \\mathcal{G}(\\mathbb{w})'"></div>
    </div>
    <div class="inline-container">
      In order to have meaningful controls here, we'll instead use a PCA base of
      the "style" space
      <div v-katex="'\\mathcal{W}'"></div>
      using
      <div v-katex="'\\mathbb{w_{base}} = Vx_{base}'"></div>
    </div>
  </div>
  <div>
    <v-slider
      v-for="index in nSliders"
      :key="index"
      :min="-slidersRange"
      :max="slidersRange"
      :step="0.01 * slidersRange * 5"
      v-model="sliderValues[index - 1]"
      thumb-label="always"
    >
      <template v-slot:label>
        <div v-katex="'x_{' + (index - 1) + '}'"></div>
      </template>
    </v-slider>
    <!-- Buttons -->
    <v-btn @click="setSlidersToZero">Set Sliders to 0</v-btn>
    <v-btn @click="setSlidersToRandom">Set Sliders to Random</v-btn>
    <v-btn @click="setOthersToZero">Set Others to 0</v-btn>
    <v-btn @click="setOthersToRandom">Set Others to Random</v-btn>
  </div>
</template>

<script>
import "katex/dist/katex.min.css";

export default {
  props: {
    nSliders: {
      type: Number,
      default: 10,
    },
    slidersRange: {
      type: Number,
      default: 1.0,
    },
  },
  data() {
    return {
      sliderValues: Array(this.nSliders).fill(0),
    };
  },
  watch: {
    sliderValues: {
      handler(newValues) {
        this.$emit("originalSliderChange", newValues);
      },
      deep: true,
    },
  },

  methods: {
    setSlidersToZero() {
      this.sliderValues.fill(0);
    },
    setSlidersToRandom() {
      this.sliderValues = this.sliderValues.map(
        () => Math.random() * (this.slidersRange * 2) - this.slidersRange,
      );
    },
    setOthersToZero() {
      this.$emit("othersToZero");
    },
    setOthersToRandom() {
      this.$emit("othersToRandom");
    },
  },
};
</script>

<style>
.original-def {
  margin-bottom: 35px;
}
.inline-container > * {
  display: inline-block;
  margin-right: 0px; /* Adjust spacing as needed */
}

.inline-container > p {
  margin-bottom: 0; /* Remove bottom margin from paragraph if needed */
}
</style>
