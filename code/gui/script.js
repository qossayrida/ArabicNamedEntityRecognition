document.getElementById("processButton").addEventListener("click", () => {
    const textInput = document.getElementById("textInput").value.trim();
    const modelSelect = document.getElementById("modelSelect").value;
    const resultsDiv = document.getElementById("results");

    if (!textInput) {
        resultsDiv.textContent = "Please enter some text!";
        return;
    }

    // Send data to FastAPI backend
    fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            text: textInput,
            model: modelSelect,
        }),
    })
        .then((response) => response.json())
        .then((data) => {
            // Clear previous results
            resultsDiv.innerHTML = "";

            // Check if entities are returned
            if (data.entities && data.entities.length > 0) {
                // Create a table
                const table = document.createElement("table");
                table.className = "results-table";

                // Add table header
                const headerRow = document.createElement("tr");
                ["Entity", "Value"].forEach((headerText) => {
                    const th = document.createElement("th");
                    th.textContent = headerText;
                    headerRow.appendChild(th);
                });
                table.appendChild(headerRow);

                // Add rows for each entity
                data.entities.forEach((entity) => {
                    const row = document.createElement("tr");
                    const entityCell = document.createElement("td");
                    const valueCell = document.createElement("td");

                    entityCell.textContent = entity.entity;
                    valueCell.textContent = entity.value;

                    row.appendChild(entityCell);
                    row.appendChild(valueCell);
                    table.appendChild(row);
                });

                resultsDiv.appendChild(table);
            } else {
                resultsDiv.textContent = "No entities found!";
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            resultsDiv.textContent = "Error processing the request.";
        });
});
