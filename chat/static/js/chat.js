function addMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender);

    const textSpan = document.createElement('span');
    textSpan.classList.add('text');
    textSpan.innerText = text;

    messageDiv.appendChild(textSpan);
    chatMessages.appendChild(messageDiv);

    // Scroll to the bottom of the chat
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function sendMessage() {
    const userInput = document.getElementById('userInput');
    const userMessage = userInput.value;

    if (userMessage.trim() === '') return;

    // Add user message to chat
    addMessage(userMessage, 'user');

    // Clear input field
    userInput.value = '';

    setTimeout(async () => {
        const botResponse = await getBotResponse(userMessage);
        addMessage(botResponse, 'bot');
    }, 500);
}

async function clearChat() {
    try {
        const response = await fetch('/api/chat/clear-chat/', {
            method: 'DELETE',
        });

        const data = await response.json();
        return data.bot_message
    } catch (error) {
        console.error("Error deleting history:", error);
        addMessage("Sorry, an error occurred while processing your request.", 'bot');
    }
}

async function getBotResponse(message) {
    try {
        const response = await fetch('/api/chat/chat-with-gpt/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(message)
        });

        const data = await response.json();
        console.log(data)
        return data.bot_message
    } catch (error) {
        console.error("Error fetching chatbot response:", error);
        addMessage("Sorry, an error occurred while processing your request.", 'bot');
    }
}

document.getElementById("userInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
    }
});