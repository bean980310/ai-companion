import axios, { type AxiosInstance, type AxiosRequestConfig } from 'axios';

// Base URL for the backend API
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:7860';

// Create axios instance with default config
const apiClient: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add any auth tokens or common headers here
    const apiKey = localStorage.getItem('api_key');
    if (apiKey) {
      config.headers['X-API-Key'] = apiKey;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle common errors
    if (error.response) {
      switch (error.response.status) {
        case 401:
          console.error('Unauthorized');
          break;
        case 403:
          console.error('Forbidden');
          break;
        case 404:
          console.error('Not Found');
          break;
        case 500:
          console.error('Server Error');
          break;
      }
    }
    return Promise.reject(error);
  }
);

// Generic request function
export async function apiRequest<T>(
  config: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.request<T>(config);
  return response.data;
}

// Gradio API response format
interface GradioResponse<T = unknown> {
  data: T[];
}

/**
 * Call a Gradio API endpoint.
 * Gradio expects: POST /api/{api_name} with body { "data": [arg1, arg2, ...] }
 * Gradio returns: { "data": [result1, ...] }
 */
export async function gradioPredict<T = string>(
  apiName: string,
  args: unknown[],
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.post<GradioResponse<T>>(
    `/api/${apiName}`,
    { data: args },
    {
      timeout: 120000, // LLM responses can take a while
      ...config,
    }
  );
  return response.data.data[0];
}

// Helper functions for common HTTP methods
export const api = {
  get: <T>(url: string, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'GET', url }),

  post: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'POST', url, data }),

  put: <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'PUT', url, data }),

  delete: <T>(url: string, config?: AxiosRequestConfig) =>
    apiRequest<T>({ ...config, method: 'DELETE', url }),
};

export { apiClient };
export default apiClient;
