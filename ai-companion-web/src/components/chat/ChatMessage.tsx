import { User, Bot } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { ChatMessage as ChatMessageType } from '../../types';

interface ChatMessageProps {
  message: ChatMessageType;
  isStreaming?: boolean;
}

export function ChatMessage({ message, isStreaming }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isSystem = message.role === 'system';
  const content = typeof message.content === 'string' ? message.content : '';

  if (isSystem) {
    return (
      <div className="px-4 py-2 mx-4 my-2 bg-bg-tertiary/50 rounded-lg text-text-muted text-sm italic">
        {content}
      </div>
    );
  }

  return (
    <div className={`flex gap-3 px-4 py-4 ${isUser ? 'bg-transparent' : 'bg-bg-secondary/30'}`}>
      {/* Avatar */}
      <div
        className={`
          shrink-0 w-8 h-8 rounded-lg flex items-center justify-center
          ${isUser ? 'bg-accent-primary' : 'bg-accent-secondary'}
        `}
      >
        {isUser ? (
          <User size={18} className="text-white" />
        ) : (
          <Bot size={18} className="text-white" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        <div className="text-sm font-medium text-text-secondary mb-1">
          {isUser ? 'You' : 'AI Assistant'}
        </div>
        <div className="prose prose-invert prose-sm max-w-none">
          <ReactMarkdown
            components={{
              code({ node, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const isInline = !match;

                return isInline ? (
                  <code
                    className="px-1.5 py-0.5 bg-bg-tertiary rounded text-accent-primary text-sm"
                    {...props}
                  >
                    {children}
                  </code>
                ) : (
                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    className="rounded-lg !bg-bg-tertiary !mt-2 !mb-2"
                  >
                    {String(children).replace(/\n$/, '')}
                  </SyntaxHighlighter>
                );
              },
              p({ children }) {
                return <p className="mb-2 last:mb-0 text-text-primary">{children}</p>;
              },
              ul({ children }) {
                return <ul className="list-disc list-inside mb-2 text-text-primary">{children}</ul>;
              },
              ol({ children }) {
                return <ol className="list-decimal list-inside mb-2 text-text-primary">{children}</ol>;
              },
              a({ href, children }) {
                return (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent-primary hover:underline"
                  >
                    {children}
                  </a>
                );
              },
            }}
          >
            {content}
          </ReactMarkdown>
          {isStreaming && (
            <span className="inline-block w-2 h-4 bg-accent-primary animate-pulse ml-1" />
          )}
        </div>
      </div>
    </div>
  );
}

export default ChatMessage;
