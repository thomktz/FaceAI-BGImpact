<template>
  <v-container class="fill-height" fluid>
    <v-row>

      <!-- Project Description -->
      <v-col cols="12" md="6">
        <div class="project-description">
          <h2>FaceAI - Background Impact</h2>
          <p>This project implements various Generative AI models to generate faces:</p>
          <ul>
            <li>Deep Convolutional Generative Adversarial Network (DCGAN)</li>
            <li>Progressive-growing StyleGAN</li>
            <li>Variational Autoencoder (VAE)</li>
          </ul>
          <p>The projects also implements two new versions of the Flicker-Face-HQ (FFHQ)...</p>
          <!-- Add more content as needed -->
        </div>
      </v-col>

      <!-- Team Members List -->
      <v-col cols="12" md="6">
        <div class="team-members">
          <div class="member">
            <span>Thomas Kientz</span>
            <a href="https://linkedin.com/in/member1" target="_blank">
              <Icon icon="mdi:linkedin" color="#4361ee" />
            </a>
            <a href="https://github.com/member1" target="_blank">
              <Icon icon="mdi:github" color="#333" />
            </a>
          </div>
          <hr class="separator" />
          <div class="member">
            <span>Elena Loumagne</span>
            <a href="https://linkedin.com/in/member2" target="_blank">
              <Icon icon="mdi:linkedin" color="#4361ee" />
            </a>
            <a href="https://github.com/member2" target="_blank">
              <Icon icon="mdi:github" color="#333" />
            </a>
          </div>
          <hr class="separator" />
          <div class="member">
            <span>Adèle Moreau</span>
            <a href="https://linkedin.com/in/member3" target="_blank">
              <Icon icon="mdi:linkedin" color="#4361ee" />
            </a>
            <a href="https://github.com/member3" target="_blank">
              <Icon icon="mdi:github" color="#333" />
            </a>
          </div>
        </div>
      </v-col>

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
.project-description {
  
}

.team-members {
  /* Styles for team members list */
}

.member {
  display: flex;
  align-items: center;
  margin-bottom: 20px; /* Adjust the spacing as needed */
}

.member span {
  margin-right: 10px;
}

a {
  margin-right: 10px;
}

a:last-child {
  margin-right: 0;
}

.separator {
  border: none;
  height: 1px;
  background-color: #ddd; /* Light gray color; adjust as needed */
  margin: 20px 0; /* Spacing above and below the separator */
}
</style>
