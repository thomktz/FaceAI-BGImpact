<template>
  <div class="edit-controls">
    <div class="edit-def">
      <h2>Creating Edits</h2>
      <div class="inline-container">
        We can mix eigenvectors and layers to create new, more interpretable
        vectors, "edits". <br />
      </div>
    </div>

    <!-- Name of the edit -->
    <v-text-field
      label="Name of the edit"
      v-model="editName"
      @update:model-value="updateDescription"
    ></v-text-field>

    <!-- Display the current selected vectors -->
    <div class="selected-vectors-display">
      <div v-katex="selectedVectors"></div>
    </div>

    <!-- Lambda slider -->
    <div class="lambda-slider-container">
      <v-slider
        class="lambda-slider"
        v-model="lambda"
        min="-2"
        max="2"
        step="0.1"
        thumb-label="always"
        hide-details
      >
        <template v-slot:label>
          <div v-katex="'\\lambda'"></div>
        </template>
      </v-slider>
      <v-btn
        class="lambda-reset-button"
        @click="lambda = 1"
        :disabled="lambda === 1"
      >
        Reset lambda
      </v-btn>
    </div>
  </div>
</template>

<script>
import "katex/dist/katex.min.css";
export default {
  emits: ["lambda-change"],
  data() {
    return {
      editName: "",
      editMode: false,
      lambda: 1,
      selectedVectors: "",
    };
  },
  props: {
    layerList: Array,
    newSliderValues: Array,
  },
  watch: {
    layerList: {
      handler(newLayerList) {
        this.handleLayerListChange(newLayerList);
      },
      deep: true,
    },
    newSliderValues: {
      handler(newValues) {
        this.handleNewSliderValuesChange(newValues);
      },
      deep: true,
    },
    lambda(newValue) {
      this.$emit("lambda-change", newValue);
    },
  },
  methods: {
    handleLayerListChange(newLayerList) {
      // Perform actions when layerList changes
      this.updateDescription();
    },
    handleNewSliderValuesChange(newValues) {
      // Perform actions when newSliderValues changes
      this.updateDescription();
    },

    updateDescription() {
      this.selectedVectors = "";
      for (let i = 0; i < this.newSliderValues.length; i++) {
        // If the slider is not 0
        if (this.newSliderValues[i] !== 0) {
          this.selectedVectors += `${this.newSliderValues[i].toFixed(
            2,
          )} \\cdot x_{${i}, [${this.layerList[i][0]}, ${
            this.layerList[i][this.layerList[i].length - 1]
          }]} + `;
        }
      }
      // If nothing was added, do not display anything
      if (this.selectedVectors === "") {
        this.selectedVectors = "0";
      }
      // Remove the last plus sign
      this.selectedVectors =
        "\\textbf{w}_{" +
        this.editName +
        "} = \\lambda \\cdot (" +
        this.selectedVectors.slice(0, -3) +
        ") \\cdot V";
    },
  },
};
</script>

<style>
.edit-def {
  margin-bottom: 35px;
}
.inline-container > * {
  display: inline-block;
  margin-right: 0px;
}

.inline-container > p {
  margin-bottom: 0;
}
.lambda-slider-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 40px;
}

.lambda-slider {
  flex-grow: 1;
  margin-right: 10px;
}
</style>
