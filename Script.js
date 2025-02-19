function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = Food: ${data.food}, Calories: ${data.calories} kcal;
    })
    .catch(error => console.error("Error:", error));
}