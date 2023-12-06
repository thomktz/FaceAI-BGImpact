<template>
  <!-- Relocated Layer-wise exploration content -->
  <div class="new-def">
    <h2>Layer-wise exploration</h2>
    <div class="inline-container">
      The StyleGAN Synthesis network is a series of layers, which all take the
      "style" vector as input
      <div
        v-katex="
          '\\mathcal{G}(\\mathbb{w}) = \\mathcal{G}_{L-1}(\\mathbb{w}, \\mathcal{G}_{L-2}(\\mathbb{w}, \\dots \\mathcal{G}_0(\\mathbb{w})))'
        "
      ></div>
      where
      <div v-katex="'\\mathcal{G}_i'"></div>
      is the i-th layer of the StyleGAN generator. <br />
      What we can do is use another style
      <div v-katex="'\\mathbb{w}_{offset} = w_{base} + Vx_{offset}'"></div>
      and apply it to a subset of the layers
      <div v-katex="'L'"></div>
      . For example, we can apply the offset to the first layer only, and leave
      the rest of the layers unchanged
      <div v-katex="'(L=[0])'"></div>
      <div
        v-katex="
          '\\mathcal{G}(\\textbf{w}) = \\mathcal{G}_{L-1}(\\mathbb{w}_{base}, \\mathcal{G}_{L-2}(\\mathbb{w}_{base}, \\dots \\mathcal{G}_0(\\mathbb{w}_{offset})))'
        "
      ></div>
    </div>
  </div>
  <div>
    <div v-for="index in nSliders" :key="index">
      <v-slider
        v-model="sliderValues[index - 1]"
        :min="-slidersRange"
        :max="slidersRange"
        :step="0.01 * slidersRange * 5"
        thumb-label="always"
      >
        <template v-slot:label>
          <div
            v-katex="'x_{' + (index - 1) + '}'"
            style="margin-right: 5px"
          ></div>
        </template>
      </v-slider>
      <v-range-slider
        v-model="layerRanges[index - 1]"
        :ticks="[...Array(maxLayers + 1).keys()]"
        :min="0"
        :max="9"
        :step="1"
        @update:model-value="handleLayerRangeChange(index - 1)"
        show-ticks="always"
        :tick-size="4"
      >
        <template v-slot:label>
          <div
            v-katex="'L_{' + (index - 1) + '}'"
            style="margin-right: 10px"
          ></div>
        </template>
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
      default: 5,
    },
    maxLayers: {
      type: Number,
      default: 9,
    },
    slidersRange: {
      type: Number,
      default: 1.0,
    },
  },
  data() {
    return {
      sliderValues: Array(this.nSliders).fill(0),
      layerRanges: Array(this.nSliders)
        .fill()
        .map(() => [0, this.maxLayers]),
      oldLayerRanges: Array(this.nSliders)
        .fill()
        .map(() => [0, this.maxLayers]),
      fullLayerLists: Array(this.nSliders)
        .fill()
        .map(() => [...Array(this.maxLayers + 1).keys()]),
    };
  },
  watch: {
    nSliders(newVal, oldVal) {
      if (newVal > oldVal) {
        // Handle increase in number of sliders
        this.sliderValues = [
          ...this.sliderValues,
          ...Array(newVal - oldVal).fill(0),
        ];
        this.layerRanges = [
          ...this.layerRanges,
          ...Array(newVal - oldVal)
            .fill()
            .map(() => [0, this.maxLayers]),
        ];
        this.oldLayerRanges = [
          ...this.oldLayerRanges,
          ...Array(newVal - oldVal)
            .fill()
            .map(() => [0, this.maxLayers]),
        ];
        this.fullLayerLists = [
          ...this.fullLayerLists,
          ...Array(newVal - oldVal)
            .fill()
            .map(() => [...Array(this.maxLayers + 1).keys()]),
        ];
      } else {
        // Handle decrease in number of sliders
        this.sliderValues = this.sliderValues.slice(0, newVal);
        this.layerRanges = this.layerRanges.slice(0, newVal);
        this.oldLayerRanges = this.oldLayerRanges.slice(0, newVal);
        this.fullLayerLists = this.fullLayerLists.slice(0, newVal);
      }
    },
    sliderValues: {
      handler(newValues) {
        this.$emit("newSliderChange", newValues);
      },
      deep: true,
    },
  },
  methods: {
    setSlidersToZero() {
      this.sliderValues.fill(0);
      this.layerRanges = this.layerRanges.map(() => [0, this.maxLayers]);
    },
    handleLayerRangeChange(index) {
      // Check if value has changed
      if (
        this.layerRanges[index][0] == this.oldLayerRanges[index][0] &&
        this.layerRanges[index][1] == this.oldLayerRanges[index][1]
      ) {
        return;
      }
      console.log("Layer range changed");
      this.oldLayerRanges[index] = [...this.layerRanges[index]];
      this.fullLayerLists[index] = [...Array(this.maxLayers + 1).keys()].slice(
        this.layerRanges[index][0],
        this.layerRanges[index][1] + 1,
      );

      this.$emit("layerListChange", this.fullLayerLists);
    },
  },
};
</script>

<style scoped>
.v-slider {
  margin-bottom: -25px;
  padding-right: 50px;
}
.v-range-slider {
  margin-bottom: 15px;
}
.new-def {
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
