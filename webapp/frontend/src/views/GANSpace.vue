<template>
    <v-app>
      <v-main>
        <v-container fluid>
          <v-row>
            <v-col cols="4" class="panel">
              <div class="original-def">
                <h2>Base style controls</h2>
                <div class="inline-container">
                  The StyleGAN generator transforms a 256-dim "noise" vector into a 256-dim "style" vector
                  <div v-katex="'\\mathbb{w} = \\mathcal{M}(z)'"></div>
                  where
                  <div v-katex="'\\mathcal{M}'"></div>
                  is the StyleGAN Mapping network. This "style" is then used to generate an image
                  <div v-katex="'img = \\mathcal{G}(\\mathbb{w})'"></div>
                </div>
                <div class="inline-container">
                  In order to have meaningful controls here, we'll instead use a PCA base of the "style" space
                  <div v-katex="'\\mathcal{W}'"></div>
                  using 
                  <div v-katex="'\\mathbb{w_{base}} = Vx_{base}'"></div>
                </div>
              </div>
              <SliderGroup 
                :nSliders="settings.nSlidersOriginal" 
                :slidersRange="settings.slidersRange" 
                @othersToZero="othersToZero"
                @othersToRandom="othersToRandom"
                @originalSliderChange="originalSliderChange"
              ></SliderGroup>
            </v-col>
            <v-col cols="4" class="image-column">
              <ImageDisplay :imageData="imageData"></ImageDisplay>
            </v-col>
            <v-col cols="4" class="panel">
              <div class="new-def">
                <h2>Layer-wise exploration</h2>
                <div class="inline-container">
                  The StyleGAN Synthesis network is a series of layers, which all take the "style" vector as input
                  <div v-katex="'\\mathcal{G}(\\mathbb{w}) = \\mathcal{G}_{L-1}(\\mathbb{w}, \\mathcal{G}_{L-2}(\\mathbb{w}, \\dots \\mathcal{G}_0(\\mathbb{w})))'"></div>
                  where
                  <div v-katex="'\\mathcal{G}_i'"></div>
                  is the i-th layer of the StyleGAN generator. <br>
                  What we can do is use another style 
                  <div v-katex="'\\mathbb{w}_{offset} = w_{base} + Vx_{offset}'"></div>
                  and apply it to a subset of the layers 
                  <div v-katex="'L'"></div>
                  . For example, we can apply the offset to the first layer only, and leave the rest of the layers unchanged
                  <div v-katex="'(L=[0])'"></div>
                  <div v-katex="'\\mathcal{G}(\\textbf{w}) = \\mathcal{G}_{L-1}(\\mathbb{w}_{base}, \\mathcal{G}_{L-2}(\\mathbb{w}_{base}, \\dots \\mathcal{G}_0(\\mathbb{w}_{offset})))'"></div>
                </div>
              </div>
              <SliderGroupLayers 
                :nSliders="settings.nSlidersNew" 
                :slidersRange="settings.slidersRange" 
                @newSliderChange="newSliderChange"
                @layerListChange="layerListChange"
              ></SliderGroupLayers>
            </v-col>
          </v-row>
        </v-container>
      </v-main>
      <Settings @updateSettings="handleSettingsUpdate"></Settings>
    </v-app>
  </template>
  
  <script>
  import Settings from '@/components/Settings.vue';
  import ImageDisplay from '@/components/ImageDisplay.vue';
  import SliderGroup from '@/components/SliderGroup.vue';
  import SliderGroupLayers from '@/components/SliderGroupLayers.vue';
  import { apiClient } from '@/apiConfig';
  import 'katex/dist/katex.min.css';
  
  export default {
    components: {
      Settings,
      ImageDisplay,
      SliderGroup,
      SliderGroupLayers
    },
    data: () => ({
      settings: {
        modelName: 'grey',
        slidersRange: 2.0,
        nSlidersNew: 5,
        nSlidersOriginal: 10
      },
      imageData: '',
      layerList: Array(5).fill().map(() => [...Array(10).keys()]),
      originalSliderValues: Array(10).fill(0),
      newSliderValues: Array(5).fill(0),
    }),
    mounted() {
        this.handleSettingsUpdate(this.settings);
        this.originalSliderChange(this.originalSliderValues)
        this.getImage();
    },
    methods: {
      handleSettingsUpdate(newSettings) {
        this.settings = { ...newSettings };
        this.layerList = Array(this.settings.nSlidersNew).fill().map(() => [...Array(10).keys()]);
        this.originalSliderValues = Array(this.settings.nSlidersOriginal).fill(0);
        this.newSliderValues = Array(this.settings.nSlidersNew).fill(0);
        apiClient.post("/stylegan/set-n-sliders", { n_sliders: this.settings.nSlidersOriginal })
          .then(response => {
            console.log(response);
          })
          .catch(error => {
            console.log(error);
          });
        this.getImage();

      },
      othersToZero() {
        console.log('Set others to zero');
        apiClient.get("/stylegan/zero-x1-others")
          .then(response => {
            this.originalSliderChange(this.originalSliderValues)
            this.getImage();
          })
          .catch(error => {
            console.log(error);
          });
      },
      othersToRandom() {
        console.log('Set others to random');
        apiClient.get("/stylegan/random-x1-others")
          .then(response => {
            this.originalSliderChange(this.originalSliderValues)
            this.getImage();
          })
          .catch(error => {
            console.log(error);
          });
      },
      newSliderChange(newValues) {
        this.newSliderValues = newValues;
        this.getImage();
      },
      originalSliderChange(newValues) {
        this.originalSliderValues = newValues;
        apiClient.post('/stylegan/x1-sliders', { slider_values: newValues })
          .then(response => {
            this.getImage();
          })
          .catch(error => {
            console.log(error);
          });
      },
      layerListChange(newRanges) {
        this.layerList = newRanges;
        this.getImage();
      },
      getImage() {
        apiClient.post('/stylegan/generate-image', {
            "eigenvector_strengths": this.newSliderValues,
            "layers_list": this.layerList
          })
          .then(response => {
            this.imageData = `data:image/jpeg;base64,${response.data.image}`;
          })
          .catch(error => {
            console.log(error);
          });
      }
    }
  }
  </script>
  
  <style>
  .inline-container > * {
  display: inline-block;
  margin-right: 0px; /* Adjust spacing as needed */
}

.inline-container > p {
  margin-bottom: 0; /* Remove bottom margin from paragraph if needed */
}
.original-def {
  margin-bottom: 35px;
}
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
.panel {
  z-index: 1000;
}
  </style>
  ```