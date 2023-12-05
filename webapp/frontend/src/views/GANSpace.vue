<template>
    <v-app>
      <v-main>
        <v-container fluid>
          <v-row>
            <v-col cols="4">
              <SliderGroup 
                :nSliders="settings.nSlidersOriginal" 
                :slidersRange="settings.slidersRange" 
                @othersToZero="othersToZero"
                @othersToRandom="othersToRandom"
                @originalSliderChange="originalSliderChange"
              ></SliderGroup>
            </v-col>
            <v-col cols="4">
              <ImageDisplay :imageData="imageData"></ImageDisplay>
            </v-col>
            <v-col cols="4">
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
      layerList: Array(5).fill().map(() => [0, 1, 2, 3, 4]),
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
        this.layerList = Array(this.settings.nSlidersNew).fill().map(() => [0, 1, 2, 3, 4]);
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
  