import { api } from './client';
import type { Session, ChatMessage, ApiResponse } from '../types';

// Get all sessions
export async function getSessions(): Promise<ApiResponse<Session[]>> {
  return api.get('/api/sessions');
}

// Create new session
export async function createSession(name?: string): Promise<ApiResponse<Session>> {
  return api.post('/api/sessions', { name });
}

// Get session by ID
export async function getSession(sessionId: string): Promise<ApiResponse<Session>> {
  return api.get(`/api/sessions/${sessionId}`);
}

// Update session (rename)
export async function updateSession(
  sessionId: string,
  name: string
): Promise<ApiResponse<Session>> {
  return api.put(`/api/sessions/${sessionId}`, { name });
}

// Delete session
export async function deleteSession(sessionId: string): Promise<ApiResponse<void>> {
  return api.delete(`/api/sessions/${sessionId}`);
}

// Delete all sessions
export async function deleteAllSessions(): Promise<ApiResponse<void>> {
  return api.delete('/api/sessions');
}

// Get chat history for a session
export async function getChatHistory(
  sessionId: string
): Promise<ApiResponse<ChatMessage[]>> {
  return api.get(`/api/sessions/${sessionId}/history`);
}

// Save chat history
export async function saveChatHistory(
  sessionId: string,
  messages: ChatMessage[]
): Promise<ApiResponse<void>> {
  return api.post(`/api/sessions/${sessionId}/history`, { messages });
}
