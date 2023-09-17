let sHC = document.getElementById("stockHistoryChart").getContext("2d");

function createChart(history_data) {
    let myChart = new Chart(sHC, {
        type: "line",
        data: {
            labels: [],
            datasets: [
            {
                data: history_data["data"]["Close"],
                backgroundColor: "rgba(153,205,1,0.6)",
            }
        ]

        }
    }
    );
        
    
}

let form = document.getElementById("ticker-input");

// form.addEventListener('submit', function(event) {
//     event.preventDefault();    // prevent page from refreshing
//     const formData = new FormData(form);  // grab the data inside the form fields
//     fetch('/get-ticker-prediction', {   
//         method: 'GET',
//         body: formData,
//     }).then(function(response) {
//         console.log(response)
//     });
// });

