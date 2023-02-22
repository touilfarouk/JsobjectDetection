// KNN Classification
// A Beginner's Guide to Machine Learning with ml5.js
// The Coding Train / Daniel Shiffman
// 1: https://youtu.be/KTNqXwkLuM4
// 2: https://youtu.be/Mwo5_bUVhlA
// 3: https://youtu.be/JWsKay58Z2g
// https://thecodingtrain.com/learning/ml5/4.1-ml5-save-load-model.html
// https://editor.p5js.org/codingtrain/sketches/RERqlwJL

let video
let features
let knn
let labelP
let ready = false

function setup() {
  createCanvas(320, 240)
  video = createCapture(VIDEO)
  video.size(320, 240)
  features = ml5.featureExtractor("MobileNet", modelReady)
  knn = ml5.KNNClassifier()
  labelP = createP("need training data")
  labelP.style("font-size", "32pt")
}

function getLabel(result) {
  const entries = Object.entries(result.confidencesByLabel)
  let greatestConfidence = entries[0]
  for (let i = 0; i < entries.length; i++) {
    if (entries[i][1] > greatestConfidence[1]) {
      greatestConfidence = entries[i]
    }
  }
  return greatestConfidence[0]
}

function goClassify() {
  const logits = features.infer(video)
  knn.classify(logits, function (error, result) {
    if (error) {
      console.error(error)
    } else {
      console.log("Classify - Result : " + JSON.stringify(result))
      labelP.html(getLabel(result))
      goClassify()
    }
  })
}

function keyPressed() {
  const logits = features.infer(video)
  if (key == "p") {
    knn.addExample(logits, "PenDrive")
    console.log("PenDrive")
  } else if (key == "b") {
    knn.addExample(logits, "Battery")
    console.log("Battery")
  } else if (key == "s") {
    knn.addExample(logits, "Screw")
    console.log("Battery")
  } else if (key == "e") {
    knn.addExample(logits, "Empty")
    console.log("empty")
  } else if (key == "c") {
    knn.save("model.json")
  }
}

function modelReady() {
  console.log("model ready!")
  // Comment back in to load your own model!
  knn.load("model.json", function () {
    console.log("knn loaded")
  })
}

function draw() {
  image(video, 0, 0)
  if (!ready && knn.getNumLabels() > 0) {
    goClassify()
    ready = true
  }
}

const saveFile = (name, data) => {
  const downloadElt = document.createElement("a")
  const blob = new Blob([data], { type: "octet/stream" })
  const url = URL.createObjectURL(blob)
  downloadElt.setAttribute("href", url)
  downloadElt.setAttribute("download", name)
  downloadElt.style.display = "none"
  document.body.appendChild(downloadElt)
  downloadElt.click()
  document.body.removeChild(downloadElt)
  URL.revokeObjectURL(url)
}
