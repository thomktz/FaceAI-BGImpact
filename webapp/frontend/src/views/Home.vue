<template>
  <v-container fluid>
    <v-row class="container">

      <!-- Project Description with a larger width -->
      <div class="project-description">
        <h2>FaceAI - Background Impact</h2>
        <p>Evaluating the effect of removing an image's background on the training of facial generation AIs.</p>
        <p>This project implements various Generative AI models to generate faces:</p>
        <ul>
          <li>Deep Convolutional Generative Adversarial Network (DCGAN)</li>
          <li>Progressive-growing StyleGAN</li>
          <li>Variational Autoencoder (VAE)</li>
        </ul>
        <p>The projects also implements two new versions of the Flicker-Face-HQ (FFHQ)...</p>
        <div class="image-placeholder">Image Placeholder</div>
      </div>

      <!-- Team Members List with a smaller width -->
      <div class="team-members">
        <div class="member">
          <span>Thomas Kientz</span>
          <a href="https://linkedin.com/in/thomaskientz" target="_blank">
            <Icon icon="mdi:linkedin" color="#4361ee" />
          </a>
          <a href="https://github.com/thomaskientz" target="_blank">
            <Icon icon="mdi:github" color="#333" />
          </a>
        </div>
        <hr class="separator" />
        <div class="member">
          <span>Elena Loumagne</span>
          <a href="https://linkedin.com/in/elenaloumagne" target="_blank">
            <Icon icon="mdi:linkedin" color="#4361ee" />
          </a>
          <a href="https://github.com/elenaloumagne" target="_blank">
            <Icon icon="mdi:github" color="#333" />
          </a>
        </div>
        <hr class="separator" />
        <div class="member">
          <span>Adèle Moreau</span>
          <a href="https://linkedin.com/in/adèlemoreau" target="_blank">
            <Icon icon="mdi:linkedin" color="#4361ee" />
          </a>
          <a href="https://github.com/adèlemoreau" target="_blank">
            <Icon icon="mdi:github" color="#333" />
          </a>
        </div>
      </div>

    </v-row>
  </v-container>
</template>

<script>
import { Icon } from '@iconify/vue';

export default {
  components: {
    Icon,
  },
  setup() {
    return {};
  },
};
</script>

<style scoped>
.v-container {
  margin-top: 20px;
  margin-left: 20px;
}

.project-description {
  padding-right: 24px;
  width: 66vw;
  min-width: 66vw;
  padding-left: 0;
  margin-left: 0;
  list-style-position: inside;
}

.image-placeholder {
  background-color: #eee;
  height: 200px;
  width: 100%;
  margin-top: 20px;
  display: flex;
  justify-content: center;
  font-style: italic;
}

.team-members {
  padding-left: 24px;
  width: 33vw;
  min-width: 33vw;
}

.member {
  display: flex;
  align-items: center;
  margin-bottom: 20px;
}

.member span {
  margin-right: 10px;
}

a {
  margin-right: 10px;
}

.v-col {
  padding-left: 0px !important; /* Remove padding from columns */
  padding-right: 0px !important; /* Remove padding from columns */
}

.separator {
  border: none;
  height: 1px;
  background-color: #ddd;
  margin: 20px 0;
}
</style>
