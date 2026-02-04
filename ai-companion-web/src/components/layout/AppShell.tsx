import { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { useSessionStore } from '../../stores/sessionStore';
import { useModelStore } from '../../stores/modelStore';

export function AppShell() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Session store
  const {
    sessions,
    currentSessionId,
    setCurrentSession,
    createSession,
    deleteSession,
  } = useSessionStore();

  // Model store
  const {
    providers,
    selectedProvider,
    setSelectedProvider,
    models,
    selectedModel,
    setSelectedModel,
    apiKey,
    setApiKey,
    showApiKey,
  } = useModelStore();

  const handleNewSession = () => {
    createSession();
    setSidebarOpen(false);
  };

  const handleSelectSession = (sessionId: string) => {
    setCurrentSession(sessionId);
    setSidebarOpen(false);
  };

  return (
    <div className="min-h-screen bg-bg-primary flex flex-col">
      <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          sessions={sessions}
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          onDeleteSession={deleteSession}
          providers={providers}
          selectedProvider={selectedProvider}
          onProviderChange={setSelectedProvider}
          models={models}
          selectedModel={selectedModel}
          onModelChange={setSelectedModel}
          apiKey={apiKey}
          onApiKeyChange={setApiKey}
          showApiKey={showApiKey}
        />

        <main className="flex-1 overflow-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
}

export default AppShell;
