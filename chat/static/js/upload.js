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

    const requestData = {
        message: "Show anomalies",
        log_data: logData
    };
    try {
        // Send request to the backend to analyze the log data
        const response = await fetch('/api/uploader/find_anomalies/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const data = await response.json();

        addMessage(data.message, 'bot')

        document.getElementById('spinner').hidden = true;

        // this.style.display = "flex";
        document.getElementById("filterAnomaliesButton").hidden = false;

    } catch (error) {
        console.error("Error fetching GPT response:", error);
    }
});