import axios from "axios";
import { useToast } from "vue-toastification";

// Create an Axios instance
const apiClient = axios.create({
  withCredentials: true,
  baseURL: process.env.VUE_APP_API_URL,
});

console.log("Using API URL:", process.env.VUE_APP_API_URL);

// Create toast instance
const toast = useToast();

// Add a request interceptor
apiClient.interceptors.request.use(
  (request) => {
    // Log the full request details here
    // console.log("Starting Request", JSON.stringify(request, null, 2));
    return request;
  },
  (error) => {
    // Do something with request error
    c; // onsole.error("Request Error:", error);
    return Promise.reject(error);
  },
);

export { apiClient, toast };
