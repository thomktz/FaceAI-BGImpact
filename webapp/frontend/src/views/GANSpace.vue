<template>
    <v-app>
      <Settings @updateSettings="handleSettingsUpdate"></Settings>
      <v-main>
        <v-container fluid>
          <v-row>
            <v-col cols="4">
              <SliderGroup :nSliders="settings.nSlidersOriginal"></SliderGroup>
            </v-col>
            <v-col cols="4">
              <ImageDisplay></ImageDisplay>
            </v-col>
            <v-col cols="4">
              <SliderGroupLayers :nSliders="settings.nSlidersNew" :maxLayers="10"></SliderGroupLayers>
            </v-col>
          </v-row>
        </v-container>
      </v-main>
    </v-app>
  </template>
  
  <script>
  import Settings from '@/components/Settings.vue';
  import ImageDisplay from '@/components/ImageDisplay.vue';
  import SliderGroup from '@/components/SliderGroup.vue';
  import SliderGroupLayers from '@/components/SliderGroupLayers.vue';
  
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
        slidersRange: 1.0,
        nSlidersNew: 5,
        nSlidersOriginal: 10
      }
    }),
    methods: {
      handleSettingsUpdate(newSettings) {
        this.settings = { ...newSettings };
      }
    }
  }
  </script>
  