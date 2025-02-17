document.getElementById('csvFileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const rows = e.target.result.split('\n').map(row => row.split(','));

        const tableHead = document.getElementById('tableHead');
        const tableBody = document.getElementById('tableBody');

        tableHead.innerHTML = '';
        tableBody.innerHTML = '';

        // Populate header
        if (rows.length > 0) {
            rows[0].forEach(header => {
                const th = document.createElement('th');
                th.textContent = header.trim();
                tableHead.appendChild(th);
            });
        }

        // Populate table rows
        rows.slice(1).forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell.trim();
                tr.appendChild(td);
            });
            tableBody.appendChild(tr);
        });
    };

    reader.readAsText(file);
});

document.getElementById('filterAnomaliesButton').addEventListener('click', async function() {

    this.style.display = "none";
    document.getElementById('spinner').style.display = "block";

    // Collect log data from table rows (excluding the header)
    const rows = Array.from(document.querySelectorAll('#tableBody tr'));
    const logData = rows.map(row => {
        const cells = row.querySelectorAll('td');
        return cells.length > 0 ? cells[0].textContent.trim() : ''; // Get only the first cell
    });

    console.log(logData);

    // Format log data for the request
    const requestData = {
        message: "Show anomalies", // You can modify this message if needed
        log_data: logData // This is the collected data
    };
    try {
        // Send request to the backend to analyze the log data
        const response = await fetch('/api/uploader/find_anomalies/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        // After the request is successful, the backend will render the chat view
        const data = await response.json();

        const responseContainer = document.getElementById('gptResponse');
        responseContainer.innerHTML = ""; // Clear previous content

        document.getElementById('spinner').style.display = "none";

        data.message.forEach(log => {
            const logDiv = document.createElement('div');
            logDiv.classList.add('log-entry'); // Optional class for styling
            logDiv.innerHTML = `<pre>${log}</pre>`; // Wrap log message in a paragraph
            responseContainer.appendChild(logDiv);
            const separator = document.createElement('hr'); // Creates a horizontal line
            responseContainer.appendChild(separator);

        });

        document.getElementById('spinner').style.display = "none";
        document.getElementById('AskQuestionBlock').style.display = "flex";


        const logString = data.logs.join(',');
        const botResponseString = data.message.join(',');

        const url = `{% url 'home' %}?list1=${encodeURIComponent(logString)}&list2=${encodeURIComponent(botResponseString)}`

        document.getElementById("ask-question-link").href = url

        console.log(document.getElementById("ask-question-link"))
        console.log(url)

    } catch (error) {
        console.error("Error fetching GPT response:", error);
    }
});