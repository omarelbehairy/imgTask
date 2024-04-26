function uploadImage() {
    var fileInput = document.getElementById('fileInput');
    var file = fileInput.files[0];
    var formData = new FormData();
    formData.append('file', file);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);
    xhr.onload = function() {
        if (xhr.status === 200) {
            console.log('Image uploaded successfully');
            // If you want to do something after the image is uploaded, you can add it here
        } else {
            console.error('Error uploading image');
        }
    };
    xhr.send(formData);
}

