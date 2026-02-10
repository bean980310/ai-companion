import { gradioPredict } from './client';
import type { ChatMessage } from '../types';

export interface GradioChatRequest {
  message: string;
  model: string;
  provider: string;
  systemMessage?: string;
  apiKey?: string;
  temperature?: number;
  maxLength?: number;
}

/**
 * Send a chat message to the Gradio backend.
 * Uses the Gradio API format: POST /api/chat with { data: [message, model, provider, ...] }
 *
 * The Gradio 'chat' endpoint (api_name="chat") expects these positional arguments:
 *   0: message (str)
 *   1: model (str)
 *   2: provider (str)
 *   3: system_message (str)
 *   4: api_key (str)
 *   5: temperature (float)
 *   6: max_length (int)
 */
export async function sendChatMessage(
  request: GradioChatRequest
): Promise<string> {
  const {
    message,
    model = 'gpt-4o',
    provider = 'openai',
    systemMessage = 'You are a helpful AI assistant.',
    apiKey = '',
    temperature = 0.7,
    maxLength = -1,
  } = request;

  return gradioPredict<string>('chat', [
    message,
    model,
    provider,
    systemMessage,
    apiKey,
    temperature,
    maxLength,
  ]);
}

// Generate chat title using the Gradio endpoint
export async function generateChatTitle(
  content: string,
  model: string = 'gpt-4o-mini',
  provider: string = 'openai'
): Promise<string> {
  return gradioPredict<string>('generate_title', [content, model, provider]);
}

// List available models
export async function listModels(
  category: 'all' | 'llm' | 'image' = 'all'
): Promise<Record<string, unknown>> {
  return gradioPredict<Record<string, unknown>>('list_models', [category]);
}

// List chat sessions
export async function listSessions(): Promise<ChatMessage[]> {
  return gradioPredict<ChatMessage[]>('list_sessions', []);
}

// Get chat history
export async function getChatHistory(sessionId: string): Promise<ChatMessage[]> {
  return gradioPredict<ChatMessage[]>('get_history', [sessionId]);
}
