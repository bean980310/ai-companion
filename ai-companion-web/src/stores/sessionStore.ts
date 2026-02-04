import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Session } from '../types';

interface SessionState {
  sessions: Session[];
  currentSessionId: string | null;
  isLoading: boolean;

  // Actions
  setSessions: (sessions: Session[]) => void;
  setCurrentSession: (sessionId: string | null) => void;
  createSession: () => void;
  deleteSession: (sessionId: string) => void;
  updateSession: (sessionId: string, updates: Partial<Session>) => void;
  setLoading: (loading: boolean) => void;
}

const generateId = () => crypto.randomUUID();

export const useSessionStore = create<SessionState>()(
  persist(
    (set, get) => ({
      sessions: [],
      currentSessionId: null,
      isLoading: false,

      setSessions: (sessions) => set({ sessions }),

      setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),

      createSession: () => {
        const newSession: Session = {
          id: generateId(),
          name: 'New Chat',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          lastActivity: new Date().toISOString(),
          lastCharacter: 'AI Assistant',
        };

        set((state) => ({
          sessions: [newSession, ...state.sessions],
          currentSessionId: newSession.id,
        }));
      },

      deleteSession: (sessionId) => {
        const { sessions, currentSessionId } = get();
        const newSessions = sessions.filter((s) => s.id !== sessionId);

        set({
          sessions: newSessions,
          currentSessionId:
            currentSessionId === sessionId
              ? newSessions[0]?.id || null
              : currentSessionId,
        });
      },

      updateSession: (sessionId, updates) => {
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? { ...s, ...updates, updatedAt: new Date().toISOString() }
              : s
          ),
        }));
      },

      setLoading: (loading) => set({ isLoading: loading }),
    }),
    {
      name: 'ai-companion-sessions',
      partialize: (state) => ({
        sessions: state.sessions,
        currentSessionId: state.currentSessionId,
      }),
    }
  )
);

export default useSessionStore;
