let canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");
    // let canvas2 = document.querySelector('#canvas2')
    const predlabel = document.querySelector('#predlabel');
    var facecrop = 0



async function dothat(){
    try{
        const mysteryVideo = document.getElementById('video')
        const camDetails = await setupWebcam(mysteryVideo)
        const model = await loadModel()
        const blazefacemodel = await loadBlazeface()
        var mychart = create_chart();
        performDetections(model, blazefacemodel, mysteryVideo, mychart)
    } catch (e) {
        console.error(e)
    }  
    
    }

    async function setupWebcam(videoRef) {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const webcamStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
            facingMode: 'user',
            width: 600, height: 400,
            },
        })

        if ('srcObject' in videoRef) {
            videoRef.srcObject = webcamStream
        } else {
            videoRef.src = window.URL.createObjectURL(webcamStream)
        }
        }
    }

    dothat()

    async function loadModel() {
    await tf.ready()
    const modelPath =
      '/static/tfjsmodel/model.json'

    return await tf.loadGraphModel(modelPath)
  }

  async function loadBlazeface(){
    return await blazeface.load()
  }

  async function performDetections(model, blazefacemodel, videoRef, mychart) {
      try{
        tf.engine().startScope()
        const faceoutput = await findFace(blazefacemodel, videoRef)

        console.log('faceoutput', faceoutput)
        const gray = convert_to_grayscale(faceoutput);

        const transformed = tf.image.resizeNearestNeighbor(gray, [48,48], true)
                                    .div(255)
                                    .reshape([1,1,48,48]);
        
        const pred = model.predict(transformed);
        const xValues = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown"];
        const predlabelp = xValues[argMax(pred)]

        const prediction = pred.dataSync()       
        await update_chart(prediction, mychart);

        tf.dispose([gray, transformed, pred, prediction])
        tf.engine().endScope()

        requestAnimationFrame(() => {
            setTimeout(() => {
                performDetections(model, blazefacemodel, videoRef, mychart)
            }, 100)
        })
      }catch (e){
        console.error(e)
        requestAnimationFrame(() => {
            setTimeout(() => {
                performDetections(model, blazefacemodel, videoRef, mychart)
            }, 100)
        })
      }
  }

  async function findFace(model, video){
    const prediction = await model.estimateFaces(video)
    ctx.drawImage(video, 0, 0, 600, 400);

    prediction.forEach((pred) => {
        
        // // draw the rectangle enclosing the face
        ctx.beginPath();
        ctx.lineWidth = "2";
        ctx.strokeStyle = "white";
        // the last two arguments are width and height
        // since blazeface returned only the coordinates, 
        // we can find the width and height by subtracting them.
        ctx.rect(
        pred.topLeft[0],
        pred.topLeft[1],
        pred.bottomRight[0] - pred.topLeft[0],
        pred.bottomRight[1] - pred.topLeft[1]
        );
        ctx.stroke()
        

        
        // drawing small rectangles for the face landmarks
        // ctx.fillStyle = "red";
        // pred.landmarks.forEach((landmark) => {
        // ctx.fillRect(landmark[0], landmark[1], 5, 5);
        // });            

        var imgtensor = tf.browser.fromPixels(video);
        
    
        var lx = parseInt(pred.topLeft[1])
        var ly = parseInt(pred.topLeft[0])
        var bx = parseInt(pred.bottomRight[1])
        var by = parseInt(pred.bottomRight[0])
        

        console.log(imgtensor.shape[0])
        console.log(imgtensor.shape[1])
        
        var sw = parseInt(bx - lx)
        var sh = parseInt(by - ly)

        if((lx < 0) || (lx > 400) || (ly < 0) || (ly > 600) || 
        (bx < 0) || (bx > 400) || (by < 0) || (by > 600)){
            lx = 0
            ly = 0
            sw = 400
            sh = 600
        }

        console.log(lx,ly, bx, by, sw, sh)
        
        facecrop =  tf.slice(imgtensor, [lx, ly, 0], [sw, sh, 3]);

        // tf.browser.toPixels(facecrop, canvas2)

        });

        return await facecrop
  }


    async function update_chart(prediction, mychart){
        const barColorList = [];
        const alpha = [1,1,1,1,1,1,1,1,1];
        const min = Math.min(...prediction)
        const max = Math.max(...prediction)
        const upper = 1
        const lower = -1

        for(var i=0; i < prediction.length; i++){
            if(prediction[i] < 0){
                var str_a = "rgba(145, 16, 53,";
            }else{
                var str_a = "rgba(64, 156, 88,";
            }
            var str_b = String(alpha[i]);
            var color = str_a + str_b + ")";
            barColorList.push(color);
            prediction[i] = (((prediction[i] - min) / (max - min)) * (upper - lower)) + lower
        } 

        mychart.data.datasets[0].backgroundColor = barColorList
        mychart.data.datasets[0].data = prediction

        mychart.update()
        

        const labels = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown"];
        predlabel.innerHTML = labels[argMax(prediction)]
       
    }

    function create_chart(){
        const xValues = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown"];
        const yValues = [0, 0, 0, 0, 0, 0, 0, 0, 0];
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

        var mychart = new Chart("chart1", {
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
        return mychart;

    }

    function convert_to_grayscale(image){
        // the scalars needed for conversion of each channel
        // per the formula: gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
        rFactor = tf.scalar(0.2989);
        gFactor = tf.scalar(0.5870);
        bFactor = tf.scalar(0.1140);

        // separate out each channel. imgtensor.shape[0] and imgtensor.shape[1] will give you
        // the correct dimensions regardless of image size
        r = image.slice([0,0,0], [image.shape[0], image.shape[1], 1]);
        g = image.slice([0,0,1], [image.shape[0], image.shape[1], 1]);
        b = image.slice([0,0,2], [image.shape[0], image.shape[1], 1]);

        // add all the tensors together, as they should all be the same dimensions.
        gray = r.mul(rFactor).add(g.mul(gFactor)).add(b.mul(bFactor));

        return gray;
    }

    function argMax(arr) {
        if (arr.length === 0) {
            return -1;
        }

        var max = arr[0];
        var maxIndex = 0;

        for (var i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                maxIndex = i;
                max = arr[i];
            }
        }

        return maxIndex;
    }
