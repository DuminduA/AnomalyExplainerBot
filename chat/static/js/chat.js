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

    // Simulate chatbot response (you'll replace this with a server call)
    setTimeout(() => {
        const botResponse = getBotResponse(userMessage);
        addMessage(botResponse, 'bot');
    }, 500);
}

async function getBotResponse(message) {
    try {
        const response = await fetch('/api/chat/get_response/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(message)
        });

        const data = await response.json();
        return data.bot_message
    } catch (error) {
        console.error("Error fetching chatbot response:", error);
        addMessage("Sorry, an error occurred while processing your request.", 'bot');
    }

}