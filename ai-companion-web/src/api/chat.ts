import { api } from './client';
import type { ChatMessage, ApiResponse } from '../types';

export interface ChatCompletionRequest {
  message: string;
  sessionId: string;
  systemMessage?: string;
  model: string;
  provider: string;
  modelType?: string;
  apiKey?: string;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
  maxLength?: number;
  enableThinking?: boolean;
  imageInput?: string;
}

export interface ChatCompletionResponse {
  content: string;
  sessionId: string;
  messageId: string;
  title?: string;
}

// Chat completion
export async function sendChatMessage(
  request: ChatCompletionRequest
): Promise<ApiResponse<ChatCompletionResponse>> {
  return api.post('/api/chat/completion', request);
}

// Stream chat completion (SSE)
export function streamChatMessage(
  request: ChatCompletionRequest,
  onMessage: (content: string) => void,
  onError: (error: Error) => void,
  onComplete: () => void
): () => void {
  const eventSource = new EventSource(
    `/api/chat/stream?${new URLSearchParams({
      message: request.message,
      sessionId: request.sessionId,
      model: request.model,
      provider: request.provider,
    })}`
  );

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.content) {
        onMessage(data.content);
      }
      if (data.done) {
        eventSource.close();
        onComplete();
      }
    } catch (error) {
      console.error('Failed to parse SSE message:', error);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    onError(new Error('Stream connection failed'));
  };

  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

// Generate chat title
export async function generateChatTitle(
  messages: ChatMessage[]
): Promise<ApiResponse<{ title: string }>> {
  return api.post('/api/chat/title', { messages });
}
