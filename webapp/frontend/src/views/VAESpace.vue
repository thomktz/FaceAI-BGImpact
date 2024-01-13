<template>
  <v-app>
    <v-main>
      <v-container fluid>
        <v-row>
          <!-- Collapsible "Base Style Controls" Panel -->
          <v-col cols="4" class="panel-column">
            <h2>Base style controls</h2>
            <SliderGroup
              :nSliders="settings.nSliders"
              :slidersRange="settings.slidersRange"
              @othersToZero="othersToZero"
              @othersToRandom="othersToRandom"
              @sliderChange="sliderChange"
            ></SliderGroup>
          </v-col>

          <!-- Image Display -->
          <v-col cols="4" class="image-column">
            <ImageDisplay :imageData="imageData"></ImageDisplay>
          </v-col>

          <!-- Edits controls -->
          <v-col cols="4" class="panel-column">
            <EditControls
              :sliderValues="sliderValues"
              @lambdaChange="lambdaChange"
            ></EditControls>
          </v-col>
        </v-row>
      </v-container>
    </v-main>
    <Settings
      @updateSettings="handleSettingsUpdate"
      @modelChange="handleModelChange"
    ></Settings>
  </v-app>
</template>

<script>
import Settings from "@/components/Settings.vue";
import ImageDisplay from "@/components/ImageDisplay.vue";
import SliderGroup from "@/components/SliderGroup.vue";
import EditControls from "@/components/EditControls.vue";
import { apiClient } from "@/apiConfig";
import "katex/dist/katex.min.css";

export default {
  components: {
    Settings,
    ImageDisplay,
    SliderGroup,
    EditControls,
  },
  data: () => ({
    settings: {
      modelName: "Grey",
      slidersRange: 2.0,
      nSliders: 10,
    },
    imageData: "",
    sliderValues: Array(5).fill(0),
    panel: null,
    lambda: 1,
    areSlidersDisabled: false,
  }),
  mounted() {
    this.handleSettingsUpdate(this.settings);
    this.sliderChange(this.sliderValues);
    this.getImage();
  },
  methods: {
    handleSettingsUpdate(newSettings) {
      this.settings = { ...newSettings };
      this.SliderValues = Array(this.settings.nSliders).fill(0);
      apiClient
        .post("/vae/set-n-sliders", {
          n_sliders: this.settings.nSliders,
          model_name: this.settings.modelName,
        })
        .then((response) => {
          console.log(response);
        })
        .catch((error) => {
          console.log(error);
        });
      this.getImage();
    },
    handleModelChange(modelName) {
      this.settings.modelName = modelName;
      this.othersToZero();
      this.getImage();
    },
    othersToZero() {
      console.log("Set others to zero");
      apiClient
        .post("/vae/zero-x1-others", {
          model_name: this.settings.modelName,
        })
        .then((response) => {
          this.sliderChange(this.SliderValues);
          this.getImage();
        })
        .catch((error) => {
          console.log(error);
        });
    },
    othersToRandom() {
      console.log("Set others to random");
      apiClient
        .post("/vae/random-x1-others", {
          model_name: this.settings.modelName,
        })
        .then((response) => {
          this.sliderChange(this.SliderValues);
          this.getImage();
        })
        .catch((error) => {
          console.log(error);
        });
    },
    sliderChange(newValues) {
      this.sliderValues = newValues;
      this.getImage();
    },
    lambdaChange(newValue) {
      this.lambda = newValue;
      this.areSlidersDisabled = newValue !== 1;
      this.getImage();
    },
    getImage() {
      // Scale the new slider values by lambda
      const scaledSliderValues = this.sliderValues.map(
        (value) => value * this.lambda,
      );
      apiClient
        .post("/vae/generate-image", {
          eigenvector_strengths: scaledSliderValues,
          model_name: this.settings.modelName,
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
