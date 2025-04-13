document.getElementById('csvFileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (!file) return;

    console.log("Uploading the file")

    window.uploadedFileName = file['name'];
    console.log("name", window.uploadedFileName)

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

function getCSRFToken() {
    let cookieValue = null;
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.startsWith('csrftoken=')) {
            cookieValue = cookie.substring('csrftoken='.length, cookie.length);
            break;
        }
    }
    return cookieValue;
}

document.getElementById('filterAnomaliesButton').addEventListener('click', async function() {

    // this.style.display = "none";
    document.getElementById("filterAnomaliesButton").hidden = true;
    document.getElementById('spinner').hidden = false;

    // Collect log data from table rows (excluding the header)
    const rows = Array.from(document.querySelectorAll('#tableBody tr'));
    const logData = rows.map(row => {
        const cells = row.querySelectorAll('td');
        return cells.length > 0 ? cells[0].textContent.trim() : ''; // Get only the first cell
    });

    console.log(logData);
    console.log("name", window.uploadedFileName);

    const fileName = window.uploadedFileName || 'NO_FILE_NAME';

    const requestData = {
        message: "Show anomalies",
        log_data: logData,
        file: fileName
    };
    try {
        // Send request to the backend to analyze the log data
        const response = await fetch('/api/uploader/find_anomalies/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();

        addMessage(data.message, 'bot')

        document.getElementById('spinner').hidden = true;
        document.getElementById("filterAnomaliesButton").hidden = false;

    } catch (error) {
        console.error("Error fetching GPT response:", error);
        addMessage("❌ An error occurred while analyzing the data. Please try again.", 'bot');
    } finally {
        document.getElementById('spinner').hidden = true;
        document.getElementById("filterAnomaliesButton").hidden = false;
    }
});