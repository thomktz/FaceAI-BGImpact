FROM node:lts-alpine

WORKDIR /app

# Install Vue CLI globally
RUN npm install -g @vue/cli@5.0.8

# Copy package.json and package-lock.json
COPY package.json .

# Install dependencies
RUN npm install

CMD ["npm", "run", "serve"]