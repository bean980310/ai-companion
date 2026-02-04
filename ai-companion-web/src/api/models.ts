import { api } from './client';
import type { Model, Character, LLMProvider, ApiResponse } from '../types';

export interface ModelsResponse {
  llm: Record<LLMProvider, string[]>;
  image: Record<string, string[]>;
}

export interface ProvidersResponse {
  llm: LLMProvider[];
  image: string[];
}

// Get all available models
export async function getModels(): Promise<ApiResponse<ModelsResponse>> {
  return api.get('/api/models');
}

// Get LLM models
export async function getLLMModels(): Promise<ApiResponse<Model[]>> {
  return api.get('/api/models/llm');
}

// Get image models
export async function getImageModels(): Promise<ApiResponse<Model[]>> {
  return api.get('/api/models/image');
}

// Get providers
export async function getProviders(): Promise<ApiResponse<ProvidersResponse>> {
  return api.get('/api/models/providers');
}

// Validate API key
export async function validateApiKey(
  provider: LLMProvider,
  apiKey: string
): Promise<ApiResponse<{ valid: boolean }>> {
  return api.post('/api/models/validate', { provider, apiKey });
}

// Get characters
export async function getCharacters(): Promise<ApiResponse<Character[]>> {
  return api.get('/api/characters');
}

// Get character by key
export async function getCharacter(key: string): Promise<ApiResponse<Character>> {
  return api.get(`/api/characters/${key}`);
}

// Get LoRAs
export async function getLoras(): Promise<ApiResponse<string[]>> {
  return api.get('/api/image/loras');
}

// Get VAEs
export async function getVaes(): Promise<ApiResponse<string[]>> {
  return api.get('/api/image/vaes');
}
