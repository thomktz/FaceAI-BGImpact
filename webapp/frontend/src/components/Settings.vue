<template>
    <v-app-bar app>
      <v-toolbar-title>GANSpace UI Settings</v-toolbar-title>
      <v-spacer></v-spacer>
      <v-text-field label="Model Name" v-model="modelName" @input="updateSettings"></v-text-field>
      <v-text-field label="Sliders Range" type="number" v-model.number="slidersRange" @input="updateSettings"></v-text-field>
      <v-text-field label="Number of New Sliders" type="number" v-model.number="nSlidersNew" @input="updateSettings"></v-text-field>
      <v-text-field label="Number of Original Sliders" type="number" v-model.number="nSlidersOriginal" @input="updateSettings"></v-text-field>
    </v-app-bar>
  </template>
  
  <script>
  export default {
    data: () => ({
      modelName: 'grey',
      slidersRange: 1.0,
      nSlidersNew: 5,
      nSlidersOriginal: 10,
    }),
    methods: {
      updateSettings() {
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
  