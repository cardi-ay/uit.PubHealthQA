
let chatHistory = [];
let isWaitingForResponse = false;

document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    const sourcesPanel = document.getElementById('sources-panel');
    const sourcesContent = document.getElementById('sources-content');
    
    addRippleEffect();
    
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        if (this.scrollHeight > 120) {
            this.style.height = '120px';
            this.style.overflowY = 'auto';
        }
    });
    
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); 
            
            const message = userInput.value.trim();
            
            if (message === '' || isWaitingForResponse) {
                return;
            }
            
            animateSend();
            
            addUserMessage(message);
            
            userInput.value = '';
            userInput.style.height = 'auto';
            
            sendMessageToServer(message);
        }
    });
    
    chatForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        
        if (message === '' || isWaitingForResponse) {
            return;
        }
        
        animateSend();
        
        addUserMessage(message);
        
        userInput.value = '';
        userInput.style.height = 'auto';
        
        sendMessageToServer(message);
    });
    
    userInput.focus();
    
    chatMessages.addEventListener('wheel', function(e) {
        e.stopPropagation();
    });
    
    applyMessageTilt();
});

function addRippleEffect() {
    const interactiveElements = document.querySelectorAll('button, .suggestion-item');
    
    interactiveElements.forEach(element => {
        element.classList.add('ripple');
        
        element.addEventListener('click', function(e) {
            const rect = element.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const ripple = document.createElement('span');
            ripple.classList.add('ripple-effect');
            ripple.style.left = `${x}px`;
            ripple.style.top = `${y}px`;
            
            element.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
}

function animateSend() {
    const sendButton = document.querySelector('form button');
    sendButton.classList.add('animate-send');
    
    // Tạo hiệu ứng rung nhẹ
    document.querySelector('.input-group').classList.add('pulse-once');
    
    setTimeout(() => {
        sendButton.classList.remove('animate-send');
        document.querySelector('.input-group').classList.remove('pulse-once');
    }, 500);
}


function applyMessageTilt() {
    const messages = document.querySelectorAll('.message-content');
    
    messages.forEach(message => {
        message.addEventListener('mousemove', function(e) {
            const rect = message.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const deltaX = (x - centerX) / centerX * 5; 
            const deltaY = (y - centerY) / centerY * 5;
            
            message.style.transform = `perspective(1000px) rotateX(${-deltaY}deg) rotateY(${deltaX}deg) scale3d(1.02, 1.02, 1.02)`;
        });
        
        message.addEventListener('mouseleave', function() {
            message.style.transform = 'perspective(1000px) rotateX(0) rotateY(0) scale3d(1, 1, 1)';
        });
    });
}

/**
 * @param {HTMLElement} element
 */
function scrollToBottom(element) {
    setTimeout(() => {
        element.scrollTop = element.scrollHeight;
    }, 50);
    
    setTimeout(() => {
        element.scrollTo({
            top: element.scrollHeight,
            behavior: 'smooth'
        });
    }, 100);
}

/**
 * @param {string} message 
 */
function addUserMessage(message) {
    const chatMessages = document.getElementById('chat-messages');
    
    const messageElement = document.createElement('div');
    messageElement.className = 'message user-message';
    
    messageElement.innerHTML = `
        <div class="message-content">
            <div class="message-avatar">
                <i class="bi bi-person"></i>
            </div>
            <div class="message-text">
                <p>${escapeHtml(message)}</p>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(messageElement);
    
    setTimeout(() => {
        applyMessageTilt();
    }, 100);
    
    scrollToBottom(chatMessages);
    
    chatHistory.push({role: 'user', content: message});
    
    showTypingIndicator();
    
    document.getElementById('sources-panel').style.display = 'none';
    
    isWaitingForResponse = true;
}


function showTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator.style.display = 'block';
    typingIndicator.style.opacity = '0';
    
    setTimeout(() => {
        typingIndicator.style.transition = 'opacity 0.3s ease';
        typingIndicator.style.opacity = '1';
    }, 10);
}

/**
 * @param {string} message - Nội dung tin nhắn
 * @param {Array} sources - Nguồn tham khảo
 */
function addBotMessage(message, sources = []) {
    const chatMessages = document.getElementById('chat-messages');
    const typingIndicator = document.getElementById('typing-indicator');
    
    // Hiệu ứng fade out cho typing indicator
    typingIndicator.style.opacity = '0';
    
    setTimeout(() => {
        // Ẩn đang nhập
        typingIndicator.style.display = 'none';
        
        // Tạo phần tử HTML cho tin nhắn
        const messageElement = document.createElement('div');
        messageElement.className = 'message bot-message';
        
        // Xử lý markdown trong tin nhắn (sử dụng thư viện marked.js)
        const markedMessage = marked.parse(message);
        
        // Tạo nội dung tin nhắn
        messageElement.innerHTML = `
            <div class="message-content">
                <div class="message-avatar">
                    <i class="bi bi-robot"></i>
                </div>
                <div class="message-text">
                    ${markedMessage}
                </div>
            </div>
        `;
        
        // Thêm tin nhắn vào khung chat
        chatMessages.appendChild(messageElement);
        
        // Thêm hiệu ứng tilt cho tin nhắn mới
        setTimeout(() => {
            applyMessageTilt();
        }, 100);
        
        // Cập nhật lịch sử chat
        chatHistory.push({role: 'assistant', content: message});
        
        // Hiển thị nguồn tham khảo nếu có
        if (sources && sources.length > 0) {
            displaySources(sources);
        }
        
        // Cập nhật trạng thái
        isWaitingForResponse = false;
        
        // Cuộn xuống tin nhắn mới nhất - cải tiến
        scrollToBottom(chatMessages);
        
        // Thêm cuộn lần thứ hai sau khi các hình ảnh có thể đã được tải
        setTimeout(() => {
            scrollToBottom(chatMessages);
        }, 500);
    }, 300);
}

/**
 * Hiển thị nguồn tham khảo
 * @param {Array} sources - Danh sách nguồn tham khảo
 */
function displaySources(sources) {
    const sourcesPanel = document.getElementById('sources-panel');
    const sourcesContent = document.getElementById('sources-content');
    
    // Xóa nội dung cũ
    sourcesContent.innerHTML = '';
    
    // Thêm từng nguồn vào panel
    sources.forEach((source, index) => {
        const sourceElement = document.createElement('div');
        sourceElement.className = 'source-item';
        
        // Xử lý metadata để hiển thị thông tin nguồn phù hợp
        let sourceMetaText = '';
        if (source.metadata) {
            if (source.metadata.title) {
                sourceMetaText += `<strong>${escapeHtml(source.metadata.title)}</strong>`;
            }
            if (source.metadata.law_id) {
                sourceMetaText += ` - ${escapeHtml(source.metadata.law_id)}`;
            }
        }
        
        sourceElement.innerHTML = `
            <div>
                <div class="mb-1">${sourceMetaText || 'Nguồn không xác định'}</div>
                <div class="small text-muted">${escapeHtml(source.content)}</div>
                <div class="small mt-1"><span class="badge bg-primary">Điểm tương đồng: ${(source.similarity * 100).toFixed(1)}%</span></div>
            </div>
        `;
        
        sourcesContent.appendChild(sourceElement);
    });
    
    // Hiển thị panel nguồn với hiệu ứng
    sourcesPanel.style.display = 'block';
    sourcesPanel.style.opacity = '0';
    
    setTimeout(() => {
        sourcesPanel.style.transition = 'opacity 0.5s ease';
        sourcesPanel.style.opacity = '1';
    }, 10);
}

/**
 * Gửi tin nhắn đến server và xử lý phản hồi
 * @param {string} message - Nội dung tin nhắn
 */
async function sendMessageToServer(message) {
    try {
        // Chuẩn bị dữ liệu gửi đi
        const data = {
            message: message,
            history: chatHistory.slice(0, -1) // Không gửi tin nhắn vừa thêm vào
        };
        
        // Gọi API
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        // Kiểm tra phản hồi
        if (!response.ok) {
            throw new Error(`Lỗi kết nối: ${response.status}`);
        }
        
        // Xử lý dữ liệu phản hồi
        const responseData = await response.json();
        
        // Thêm tin nhắn của bot vào khung chat
        addBotMessage(responseData.answer, responseData.sources);
        
    } catch (error) {
        console.error('Lỗi:', error);
        
        // Ẩn đang nhập
        document.getElementById('typing-indicator').style.display = 'none';
        
        // Hiển thị thông báo lỗi
        addBotMessage('Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau.');
        
        // Cập nhật trạng thái
        isWaitingForResponse = false;
    }
}

/**
 * Thêm gợi ý vào ô nhập liệu
 * @param {HTMLElement} element - Phần tử chứa gợi ý
 */
function addSuggestion(element) {
    const userInput = document.getElementById('user-input');
    
    // Thêm hiệu ứng khi chọn gợi ý
    element.classList.add('active-suggestion');
    setTimeout(() => {
        element.classList.remove('active-suggestion');
    }, 300);
    
    userInput.value = element.innerText.trim();
    userInput.focus();
    
    // Điều chỉnh chiều cao của textarea
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
}

/**
 * Escape HTML để ngăn chặn XSS
 * @param {string} html - Chuỗi cần escape
 * @return {string} Chuỗi đã được escape
 */
function escapeHtml(html) {
    const div = document.createElement('div');
    div.textContent = html;
    return div.innerHTML;
} 