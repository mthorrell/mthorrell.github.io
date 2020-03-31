model_loaded = "__"
data_loaded = "__"
model = load_model()
data = load_data()
table = build_table()

function update_loading_indicator(){
  el = document.getElementById("loading_indicator")
  el.innerHTML = ("Loading: <br>"
                  + data_loaded
                  + " Pet Data <br>"
                  + model_loaded
                  + " Facial Recognition Model"
                  )
}


async function build_table(){
  tbl = []
  parent = document.getElementById("results")
  
  data = await data
  
  for (var i = 0; i < data.length; i++) {
  
    tbl.push([
      document.createElement("img"),
      document.createElement("img"),
      document.createElement("div")
    ])
    
    divhere = document.createElement("tr")
    //divhere.setAttribute("class","row")
    
    div1 = document.createElement("td")
    div2 = document.createElement("td")
    div3 = document.createElement("td")
    
    //div1.setAttribute("class","column")
    //div2.setAttribute("class","column")
    //div3.setAttribute("class","column")
    
    divhere.appendChild(div1)
    divhere.appendChild(div2)
    divhere.appendChild(div3)
    
    div1.appendChild(tbl[i][0])
    div2.appendChild(tbl[i][1])
    div3.appendChild(tbl[i][2])
    
    parent.appendChild(divhere)
    
    
    //parent.appendChild(tbl[i][0])
    //parent.appendChild(tbl[i][1])
    //parent.appendChild(tbl[i][2])
    //parent.appendChild(document.createElement("br"))
    
  }
  
  return tbl  
}
async function load_model() {
  const result = await tf.loadLayersModel('./targetdir3/model.json');
  model_loaded = "&#9989;"
  update_loading_indicator()
  //run_program()
  return result
}
async function load_data(){
  var dat_array = []
  await d3.csv("tf_pet_embedding.csv").then(function(data) {
    dat_array.push(data)
  });
  data_loaded = "&#9989;"
  update_loading_indicator()
  //run_program()
  return dat_array[0]
}
async function find_distances(pets,img_emb){
  var dist_array = []
  pets.forEach(el => dist_array.push(emb_compares(img_emb,el)))
  return dist_array
}
async function show_gray(x,canvas) {
  var result = await tf.browser.toPixels(x)
  var ctx = canvas.getContext('2d')
  var idata = new ImageData(result,20,20)
  
  ctx.putImageData(idata,0,0)
  return result
}
function emb_compares(img_emb,pet_emb){
  var distance = (
    Math.pow(pet_emb[0] - img_emb[0],2) + 
    Math.pow(pet_emb[1] - img_emb[1],2) + 
    Math.pow(pet_emb[2] - img_emb[2],2) + 
    Math.pow(pet_emb[3] - img_emb[3],2) + 
    Math.pow(pet_emb[4] - img_emb[4],2)
  )
  return distance
}
function tensor_grayscale(x) {
  rFactor = tf.scalar(0.299);  // copies the values from torch
  gFactor = tf.scalar(0.587);
  bFactor = tf.scalar(0.114);

// separate out each channel. x.shape[0] and x.shape[1] will give you
// the correct dimensions regardless of image size
  r = x.slice([0,0,0], [x.shape[0], x.shape[1], 1]);
  g = x.slice([0,0,1], [x.shape[0], x.shape[1], 1]);
  b = x.slice([0,0,2], [x.shape[0], x.shape[1], 1]);

  // add all the tensors together, as they should all be the same dimensions.
  gray = r.mul(rFactor).add(g.mul(gFactor)).add(b.mul(bFactor));
  gray = gray.mul(tf.scalar(1.0/255.0))
  
  return gray
}



var div = document.getElementById("cvhere")
var img_sml = document.createElement("img")

var img_hidden = document.createElement("img")
var canvas = document.createElement("canvas")

img_hidden.onload = function(){
  imageIsLoaded()
}

document.getElementById('inputhere').onchange = function (e) {
  loadImage(
    e.target.files[0],
    function (img) {
      img.toBlob(function(blob) {
        urlhere = URL.createObjectURL(blob)
        img_sml.src = urlhere
        img_sml.width=100
        img_hidden.src = urlhere
        div.appendChild(img_sml)
      })
    },
    { noRevoke: true,
      orientation: true
    }
  )
  
}





img_input = []
img_pets = []
async function imageIsLoaded() { 
  console.log("image got uploaded")
  run_program()
}

async function run_program(){
  console.log("Required resources have been loaded. Running...")
  document.getElementById("loading_indicator").innerHTML = "Running!0"
  
  console.log("Converting to tensor of the right form")
  
  canvas.width = 20
  canvas.height = 20
  var ctx = canvas.getContext('2d')
  ctx.drawImage(img_hidden, 0, 0, 20, 20);
  
  document.getElementById("loading_indicator").innerHTML = "Running!1"
  
  img_in = tf.browser.fromPixels(canvas)
  timg = tensor_grayscale(tf.browser.fromPixels(canvas))
  console.log("Posting converted image")
  //show_gray(timg,canvas,ctx)
  
  document.getElementById("loading_indicator").innerHTML = "Running!2"
  
  console.log("Get Loaded Resources")
  var mdl = await model
  var pets  = await data
  var tbl = await table
  
  document.getElementById("loading_indicator").innerHTML = "Running!3"
  
  console.log("PRINTING PETS")
  console.log(pets)
  
  console.log("reshaping input")
  timg = timg.reshape([1,20,20,1])
  
  document.getElementById("loading_indicator").innerHTML = "Running!4"
  
  console.log("converting to vector")
  try{
    img_emb = mdl.predict(timg).arraySync()[0]
  } catch(err) {
    document.getElementById("loading_indicator").innerHTML = err.message;
  }
  
  
  //document.getElementById("loading_indicator").innerHTML = "Running!5"
  
  console.log("comparing to pet vectors")
  op = await find_distances(pets,img_emb)
  
  document.getElementById("loading_indicator").innerHTML = "Running!6"
  
  var pets_distances = []
  for (var i = 0; i < pets.length; i++) {
    pets_distances.push({
      dist:op[i],
      name:pets[i]['names'],
      filename:pets[i]['filenames']
    })
  }
  pets_distances.sort(function(x,y){
    return d3.ascending(x.dist,y.dist)
  })
  
  document.getElementById("loading_indicator").innerHTML = "Running!7"
  
  
  document.getElementById("loading_indicator").innerHTML = ""
  
  parent = document.getElementById("rtable")
  ranking = document.getElementById("ranking")
  
  for(var i = 0; i < pets_distances.length; i++)
  {
    rowhere = tbl[i]
    
    rowhere[0].width = 200
    rowhere[0].src = img_sml.src
    
    rowhere[1].width = 200
    rowhere[1].src = pets_distances[i]['filename']
    
    rowhere[2].innerHTML = ("<center> <b> <font size=20>"
                            + pets_distances[i]['name'] 
                            + "</font> </b> <br>"
                            + "Distance: " + pets_distances[i]['dist'].toFixed(2)
                            + "</center>"
                          )
    
    //img_input.push(document.createElement("img"))
    //img_input[i].width=200
    //img_input[i].src = img_hidden.src
    //ranking.appendChild(img_input[i])
    
    //ranking.insertAdjacentHTML("beforeend","")
    
    //img_pets.push(document.createElement("img"))
    //img_pets[i].width=200
    //img_pets[i].src = pets_distances[i]['filename']
    
    //ranking.appendChild(img_pets[i])
    
    //ranking.insertAdjacentHTML("beforeend",pets_distances[i]['name'])
    
    
    //ranking.appendChild(document.createElement("br"))
  }
}




