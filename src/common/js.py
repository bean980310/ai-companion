js="""
// theme.js
// 테마 전환 스크립트
const themeToggle = document.getElementById('theme-toggle');
const currentTheme = localStorage.getItem('theme') || 'light';

if (currentTheme === 'dark') {
    document.documentElement.setAttribute('data-theme', 'dark');
}

themeToggle.addEventListener('click', () => {
    let theme = document.documentElement.getAttribute('data-theme');
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    }
});

// 메시지 전송 스크립트 (예시)
const sendButton = document.getElementById('send-button');
const messageInput = document.getElementById('message-input');
const chatMessages = document.querySelector('.chat-messages');

sendButton.addEventListener('click', () => {
    const message = messageInput.value.trim();
    if (message !== '') {
        const userMessage = document.createElement('div');
        userMessage.classList.add('user-message');
        userMessage.textContent = message;
        chatMessages.appendChild(userMessage);
        messageInput.value = '';

        // 예시로 어시스턴트 메시지 추가
        const assistantMessage = document.createElement('div');
        assistantMessage.classList.add('assistant-message');
        assistantMessage.textContent = '이것은 예시 응답입니다.';
        chatMessages.appendChild(assistantMessage);

        // 채팅 창 스크롤
        const chatWindow = document.getElementById('chat-window');
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});

// 설정 팝업 예시 스크립트
const settingsPopup = document.getElementById('settings-popup');
const closeSettingsButton = document.getElementById('close-settings');

// 예를 들어 헤더에 설정 버튼 추가하고 여기에 이벤트 리스너 추가
// 현재 예제에서는 설정 팝업을 직접 열지는 않습니다.

closeSettingsButton.addEventListener('click', () => {
    settingsPopup.classList.remove('active');
});

// 확인 다이얼로그 예시 스크립트
const confirmDialog = document.getElementById('confirm-dialog');
const confirmYes = document.getElementById('confirm-yes');
const confirmNo = document.getElementById('confirm-no');

// 예를 들어 삭제 버튼을 클릭하면 다이얼로그 열기
// 현재 예제에서는 다이얼로그를 직접 열지는 않습니다.

confirmYes.addEventListener('click', () => {
    // 실제 삭제 로직을 여기에 추가
    confirmDialog.classList.remove('active');
    alert('삭제되었습니다.');
});

confirmNo.addEventListener('click', () => {
    confirmDialog.classList.remove('active');
});
"""