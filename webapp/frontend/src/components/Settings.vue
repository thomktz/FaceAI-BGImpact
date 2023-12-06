<template>
    <v-footer app padless>
      <v-container fluid>
        <v-row align="center" class="settings-footer">
          <v-col cols="auto">
            <v-toolbar-title>GANSpace UI Settings</v-toolbar-title>
          </v-col>
  
          <v-spacer></v-spacer>
  
          <v-col cols="auto">
            <v-select label="Model Type" :items="modelTypes" v-model="modelName" @input="updateSettings" hide-details="true"></v-select>
          </v-col>
          
          <v-col cols="auto">
            <v-text-field label="Sliders Range" type="number" v-model.number="slidersRange" @input="updateSettings" class="settings-input" hide-details="true"></v-text-field>
          </v-col>
          
          <v-col cols="auto">
            <v-text-field label="Number of New Sliders" type="number" v-model.number="nSlidersNew" @input="updateSettings" class="settings-input" hide-details="true"></v-text-field>
          </v-col>
          
          <v-col cols="auto">
            <v-text-field label="Number of Original Sliders" type="number" v-model.number="nSlidersOriginal" @input="updateSettings" class="settings-input" hide-details="true"></v-text-field>
          </v-col>
        </v-row>
      </v-container>
    </v-footer>
  </template>
  
  <script>
  export default {
    data: () => ({
      modelName: 'Grey',
      slidersRange: 2.0,
      nSlidersNew: 5,
      nSlidersOriginal: 10,
      modelTypes: ['Grey', 'Blur', 'Raw'],
    }),
    methods: {
      updateSettings() {
        console.log('Settings updated');
        this.$emit('updateSettings', {
          modelName: this.modelName,
          slidersRange: this.slidersRange,
          nSlidersNew: this.nSlidersNew,
          nSlidersOriginal: this.nSlidersOriginal,
        });
      }
    }
  }
  </script>
  
  <style>
  .v-footer {
    padding-top: 0px !important;
  }
  .settings-footer {
    border-top: 1px solid rgba(0, 0, 0, 0.12); /* Add a top border to the footer */
  }
  
  .settings-input {
    max-width: 200px; /* Adjust the max width of text fields */
  }
  </style>
  