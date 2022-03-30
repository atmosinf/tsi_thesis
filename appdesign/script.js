var video = document.querySelector("#videoElement");

if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function (stream) {
      video.srcObject = stream;
    })
    .catch(function (err0r) {
      console.log("Something went wrong!");
    });
}

function create_chart(){
    const xValues = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown"];
    const yValues = [0.8, 1, -0.3, -0.5, -0.2, 0.4, 0.2, -0.1, 0.1];
    const alpha = [1,1,1,1,1,1,1,1,1];
    var barColorList = [];
    
    for(var i=0; i < yValues.length; i++){
        if(yValues[i] < 0){
            var str_a = "rgba(145, 16, 53,";
        }else{
            var str_a = "rgba(64, 156, 88,";
        }
        var str_b = String(alpha[i]);
        var color = str_a + str_b + ")";
        barColorList.push(color);
    } 

    new Chart("chart1", {
        type: "bar",
        data: {
        labels: xValues,
        datasets: [{
            backgroundColor: barColorList,
            data: yValues,
            normalized: true,
            barPercentage: 1.0,
            categoryPercentage: 1.0
        }]
        },
        options: {
        // indexAxis: 'y',
        animation: {
            duration: 10
        },
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: true,
                text: "Predicted Values"
            },
            legend: {display: false},
        },
        scales: {
            x: {
                grid: {display: false},
            },
        },
        }
    });

}

create_chart();
