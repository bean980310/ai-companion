import { useCallback } from 'react';
import { ChatWindow, ChatInput } from '../components/chat';
import { useChatStore } from '../stores/chatStore';
import { useSessionStore } from '../stores/sessionStore';
import { useModelStore } from '../stores/modelStore';
import { sendChatMessage } from '../api/chat';

export function ChatPage() {
  const {
    messages,
    isStreaming,
    streamingContent,
    addMessage,
    setStreaming,
    appendStreamContent,
    clearStreamContent,
  } = useChatStore();

  const { currentSessionId, createSession, updateSession } = useSessionStore();
  const { selectedProvider, selectedModel, apiKey } = useModelStore();
  const { systemMessage, params } = useChatStore();

  const handleSendMessage = useCallback(
    async (content: string, _images?: File[]) => {
      // Create session if none exists
      let sessionId = currentSessionId;
      if (!sessionId) {
        createSession();
        sessionId = currentSessionId;
      }

      // Add user message
      addMessage({
        id: `msg_${Date.now()}`,
        role: 'user',
        content,
        timestamp: new Date().toISOString(),
      });

      // Start streaming indicator
      setStreaming(true);
      clearStreamContent();

      try {
        // Call the Gradio backend /api/chat endpoint
        const response = await sendChatMessage({
          message: content,
          model: selectedModel,
          provider: selectedProvider,
          systemMessage: systemMessage || 'You are a helpful AI assistant.',
          apiKey: apiKey || '',
          temperature: params.temperature,
          maxLength: params.maxLength,
        });

        // Simulate streaming for smooth UX (response is already complete)
        for (let i = 0; i < response.length; i++) {
          await new Promise((resolve) => setTimeout(resolve, 5));
          appendStreamContent(response[i]);
        }

        // Add assistant message
        addMessage({
          id: `msg_${Date.now()}`,
          role: 'assistant',
          content: response,
          timestamp: new Date().toISOString(),
        });

        // Update session name if it's the first message
        if (messages.length === 0 && sessionId) {
          const title = content.slice(0, 30) + (content.length > 30 ? '...' : '');
          updateSession(sessionId, { name: title });
        }
      } catch (error) {
        console.error('Failed to send message:', error);

        const errorMessage =
          error instanceof Error
            ? `Error: ${error.message}`
            : 'Sorry, an error occurred. Please try again.';

        addMessage({
          id: `msg_${Date.now()}`,
          role: 'assistant',
          content: errorMessage,
          timestamp: new Date().toISOString(),
        });
      } finally {
        setStreaming(false);
        clearStreamContent();
      }
    },
    [
      currentSessionId,
      createSession,
      addMessage,
      setStreaming,
      appendStreamContent,
      clearStreamContent,
      updateSession,
      messages.length,
      selectedModel,
      selectedProvider,
      apiKey,
      systemMessage,
      params.temperature,
      params.maxLength,
    ]
  );

  return (
    <div className="h-full flex flex-col bg-bg-primary">
      <ChatWindow
        messages={messages}
        streamingContent={streamingContent}
        isStreaming={isStreaming}
      />
      <ChatInput
        onSend={handleSendMessage}
        isLoading={isStreaming}
        placeholder="Type your message..."
      />
    </div>
  );
}

export default ChatPage;
