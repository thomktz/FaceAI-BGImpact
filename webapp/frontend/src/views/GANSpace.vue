<template>
  <v-app>
    <v-main>
      <v-container fluid>
        <v-row>
          <!-- Collapsible "Base Style Controls" Panel -->
          <v-col cols="4" class="panel-column">
            <v-expansion-panels v-model="panel">
              <v-expansion-panel>
                <v-expansion-panel-title class="panel-title">
                  <h2>Base style controls</h2>
                </v-expansion-panel-title>
                <v-expansion-panel-text>
                  <SliderGroup
                    :nSliders="settings.nSlidersOriginal"
                    :slidersRange="settings.slidersRange"
                    @othersToZero="othersToZero"
                    @othersToRandom="othersToRandom"
                    @originalSliderChange="originalSliderChange"
                  ></SliderGroup>
                </v-expansion-panel-text>
              </v-expansion-panel>
            </v-expansion-panels>
            <slider-group-layers
              :n-sliders="settings.nSlidersNew"
              :sliders-range="settings.slidersRange"
              @new-slider-change="newSliderChange"
              @layer-list-change="layerListChange"
              :disabled="areSlidersDisabled"
            ></slider-group-layers>
          </v-col>

          <!-- Image Display -->
          <v-col cols="4" class="image-column">
            <ImageDisplay :imageData="imageData"></ImageDisplay>
          </v-col>

          <!-- Edits controls -->
          <v-col cols="4" class="panel-column">
            <EditControls
              :layerList="layerList"
              :newSliderValues="newSliderValues"
              @lambdaChange="lambdaChange"
            ></EditControls>
          </v-col>
        </v-row>
      </v-container>
    </v-main>
    <Settings @updateSettings="handleSettingsUpdate"></Settings>
  </v-app>
</template>

<script>
import Settings from "@/components/Settings.vue";
import ImageDisplay from "@/components/ImageDisplay.vue";
import SliderGroup from "@/components/SliderGroup.vue";
import SliderGroupLayers from "@/components/SliderGroupLayers.vue";
import EditControls from "@/components/EditControls.vue";
import { apiClient } from "@/apiConfig";
import "katex/dist/katex.min.css";

export default {
  components: {
    Settings,
    ImageDisplay,
    SliderGroup,
    SliderGroupLayers,
    EditControls,
  },
  data: () => ({
    settings: {
      modelName: "grey",
      slidersRange: 2.0,
      nSlidersNew: 5,
      nSlidersOriginal: 10,
    },
    imageData: "",
    layerList: Array(5)
      .fill()
      .map(() => [...Array(10).keys()]),
    originalSliderValues: Array(10).fill(0),
    newSliderValues: Array(5).fill(0),
    panel: null,
    lambda: 1,
    areSlidersDisabled: false,
  }),
  mounted() {
    this.handleSettingsUpdate(this.settings);
    this.originalSliderChange(this.originalSliderValues);
    this.getImage();
  },
  methods: {
    handleSettingsUpdate(newSettings) {
      this.settings = { ...newSettings };
      this.layerList = Array(this.settings.nSlidersNew)
        .fill()
        .map(() => [...Array(10).keys()]);
      this.originalSliderValues = Array(this.settings.nSlidersOriginal).fill(0);
      this.newSliderValues = Array(this.settings.nSlidersNew).fill(0);
      apiClient
        .post("/stylegan/set-n-sliders", {
          n_sliders: this.settings.nSlidersOriginal,
        })
        .then((response) => {
          console.log(response);
        })
        .catch((error) => {
          console.log(error);
        });
      this.getImage();
    },
    othersToZero() {
      console.log("Set others to zero");
      apiClient
        .get("/stylegan/zero-x1-others")
        .then((response) => {
          this.originalSliderChange(this.originalSliderValues);
          this.getImage();
        })
        .catch((error) => {
          console.log(error);
        });
    },
    othersToRandom() {
      console.log("Set others to random");
      apiClient
        .get("/stylegan/random-x1-others")
        .then((response) => {
          this.originalSliderChange(this.originalSliderValues);
          this.getImage();
        })
        .catch((error) => {
          console.log(error);
        });
    },
    newSliderChange(newValues) {
      this.newSliderValues = newValues;
      this.getImage();
    },
    originalSliderChange(newValues) {
      this.originalSliderValues = newValues;
      apiClient
        .post("/stylegan/x1-sliders", { slider_values: newValues })
        .then((response) => {
          this.getImage();
        })
        .catch((error) => {
          console.log(error);
        });
    },
    layerListChange(newRanges) {
      this.layerList = newRanges;
      this.getImage();
    },
    lambdaChange(newValue) {
      this.lambda = newValue;
      this.areSlidersDisabled = newValue !== 1;
      this.getImage();
    },
    getImage() {
      // Scale the new slider values by lambda
      const scaledNewSliderValues = this.newSliderValues.map(
        (value) => value * this.lambda,
      );
      apiClient
        .post("/stylegan/generate-image", {
          eigenvector_strengths: scaledNewSliderValues,
          layers_list: this.layerList,
        })
        .then((response) => {
          this.imageData = `data:image/jpeg;base64,${response.data.image}`;
        })
        .catch((error) => {
          console.log(error);
        });
    },
  },
};
</script>

<style>
.new-def {
  margin-bottom: 35px;
}
.image-column {
  background-color: grey; /* Set the background color */
  position: sticky; /* Stick to the top on scroll */
  top: 0; /* Position at the top */
  margin-top: -5px;
  overflow-y: auto; /* Allow scrolling within the column if content overflows */
  z-index: 1;
}
.panel-column {
  z-index: 500;
}
.panel-title {
  padding-left: 0 !important;
  box-shadow: none !important;
  border: none !important;
  border-bottom: 1px solid rgba(0, 0, 0, 0.12) !important;
}
.v-expansion-panel__shadow {
  box-shadow: none !important;
  border: none !important;
}
</style>
