function sendMessage() {
    var userInput = document.getElementById("user-input").value;

    // Send user input to Flask API
    fetch("/start_conversation", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ user_input: userInput })
    })
    .then(response => response.json())
    .then(data => {
        // Append response to chat output
        var chatOutput = document.getElementById("chat-output");
        var messageElement = document.createElement("p");
        messageElement.textContent = "User: " + userInput;
        chatOutput.appendChild(messageElement);
        
        var responseElement = document.createElement("p");
        responseElement.textContent = "GenerativeAI: " + data.response;
        chatOutput.appendChild(responseElement);

        // Clear user input
        document.getElementById("user-input").value = "";

        // Scroll to bottom of chat output
        chatOutput.scrollTop = chatOutput.scrollHeight;
    })
    .catch(error => console.error("Error:", error));
}





    function scrollToAboutUs() {
        var aboutUsSection = document.getElementById('about-us');
        aboutUsSection.scrollIntoView({ behavior: 'smooth' });
    }



