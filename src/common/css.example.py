css="""
/* CSS 변수 정의 */
:root {
    --color-background-primary: #f8fafc;
    --color-background-secondary: #ffffff;
    --color-text-primary: #1a1a1a;
    --color-text-secondary: #6b7280;
    --color-border: #e5e7eb;
    --color-accent: #3b82f6;
    --color-accent-hover: #2563eb;
    --color-danger-hover: #fee2e2;
    --color-danger-text: #dc2626;
    --shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    --shadow-dark: 0 1px 3px rgba(0, 0, 0, 0.2);
    --border-radius: 8px;
    --padding-small: 0.5rem;
    --padding-medium: 1rem;
    --padding-large: 1.5rem;
    
    /* 다크 모드 변수 */
    --color-background-primary-dark: #1a1a1a;
    --color-background-secondary-dark: #2d2d2d;
    --color-text-primary-dark: #ffffff;
    --color-text-secondary-dark: #9ca3af;
    --color-border-dark: #404040;
    --color-accent-dark: #60a5fa;
}

/* 공통 클래스 */
.rounded {
    border-radius: var(--border-radius);
}

.shadow {
    box-shadow: var(--shadow);
}

/* 레이아웃 컴포넌트 */
.main-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: var(--padding-large);
    background-color: var(--color-background-primary);
    min-height: 100vh;
}

.header-container, .session-container, .model-container, .chat-interface, .settings-popup, .confirm-dialog {
    background-color: var(--color-background-secondary);
    padding: var(--padding-medium);
    border-radius: var(--border-radius);
    margin-bottom: var(--padding-medium);
    box-shadow: var(--shadow);
}

.session-dropdown {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 0.5rem;
}

/* 다크 모드 스타일 */
@media (prefers-color-scheme: dark) {
    body {
        background-color: #1a1a1a !important;
        color: #ffffff;  /* 필요하다면 텍스트 색상도 지정 */
    }

    /* Gradio 최상위 컨테이너도 어둡게 */
    .gradio-container {
        background-color: #1a1a1a !important;
    }
    
    .main-container {
        background-color: var(--color-background-primary-dark);
    }
    
    .header-container, .session-container, .model-container, .chat-interface, .settings-popup, .confirm-dialog {
        background-color: var(--color-background-secondary-dark);
        border-color: var(--color-border-dark);
        box-shadow: var(--shadow-dark);
    }
    
    .main-title {
        color: var(--color-text-primary-dark);
    }
    
    .chat-window {
        background-color: var(--color-background-secondary-dark);
        border-color: var(--color-border-dark);
    }
    
    .user-message {
        background-color: #3b4252;
    }
    
    .assistant-message {
        background-color: #2e3440;
    }
    
    .system-message, .input-area {
        background-color: var(--color-background-secondary-dark);
        border-color: var(--color-border-dark);
    }
    
    .message-input {
        background-color: #1a1a1a;
        border-color: var(--color-border-dark);
        color: #ffffff;
    }
    
    .status-bar {
        background-color: var(--color-background-secondary-dark);
        color: var(--color-text-secondary-dark);
    }
}

/* 텍스트 및 버튼 스타일 */
.main-title {
    margin: 0;
    color: var(--color-text-primary);
}

.language-selector, .session-dropdown, .character-dropdown, .preset-dropdown {
    background-color: var(--color-background-secondary);
    border: 1px solid var(--color-border);
    border-radius: var(--border-radius);
    padding: var(--padding-small);
}

.icon-button, .icon-button-delete, .send-button {
    padding: var(--padding-small);
    border-radius: var(--border-radius);
    border: none;
    background-color: #f3f4f6;
    cursor: pointer;
    transition: background-color 0.2s, color 0.2s;
}

.icon-button {
    background-color: #f3f4f6;
}

.icon-button:hover {
    background-color: #e5e7eb;
}

.icon-button-delete {
    background-color: #fee2e2;
    color: var(--color-danger-text);
}

.icon-button-delete:hover {
    background-color: #7f1d1d;
    color: var(--color-danger-text);
}

.send-button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.75rem 1.5rem;
    cursor: pointer;
    transition: background-color 0.2s;
}

.send-button:hover {
    background-color: var(--color-accent-hover);
}

.status-bar {
    margin-top: 1rem;
    padding: 0.5rem;
    background-color: #f8fafc;
    border-radius: 8px;
    color: #6b7280;
    font-size: 0.875rem;
}

/* 팝업 및 다이얼로그 스타일 */
.settings-popup {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: var(--color-background-secondary);
    padding: var(--padding-large);
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.2);
    z-index: 1000;
    width: 80%;
    max-width: 800px;
    max-height: 80vh;
    overflow-y: auto;
}

.popup-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--padding-medium);
}

.close-button {
    padding: 5px 10px;
    cursor: pointer;
    background: none;
    border: none;
    font-size: 20px;
}

.settings-popup::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.5);
    z-index: -1;
}

.confirm-dialog {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: var(--color-background-secondary);
    padding: var(--padding-large);
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    z-index: 1100;
    width: 400px;
    text-align: center;
}

.confirm-dialog .buttons {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: var(--padding-medium);
}
"""