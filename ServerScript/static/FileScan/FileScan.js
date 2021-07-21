'use strict'
var scanDisplay = document.getElementById("ScanDisplay");
var fileUploadButton = document.getElementById("FileUploadButton");
var fileDisplay = document.getElementById("FileDisplay");
var scanImage = document.getElementById("ScanImage");
var scanSubmitButton = document.getElementById("ScanSubmitButton");
var fileUpload = document.getElementById("FileUpload");
var reportNotice = document.getElementById("ReportNotice");
window.uploadFiles = [];
window.scanResult = [];
window.isScanning = false;

// 文件上传按钮绑定
scanImage.onclick = fileUploadButton.onclick = function(){fileUpload.click();}

// 文件拖拽效果
window.ondragleave = window.ondrop = function(e) {
    e.preventDefault();
    fileDisplay.style.border = "none";
}
fileDisplay.ondragover = function(e){e.preventDefault();}
fileDisplay.ondragenter = window.ondragover = function(e){
    e.preventDefault();
    fileDisplay.style.border = "1px dashed black";
}
fileDisplay.ondrop = function(event){
    event.preventDefault();
    addFile(event.dataTransfer.files);
    fileDisplay.style.border = "none";
}

// 上传文件按钮设置
fileUpload.onchange = function(){

    addFile(fileUpload.files);

    // 重置
    fileUpload.value = null;
};

var addFile = function(files) {
    // 如果出现文件上传，清空放大镜
    if(uploadFiles.length == 0 && fileDisplay.children.length != 0) fileDisplay.removeChild(scanImage);

    for(var file of files) {
        // 重复则跳过
        var flag = false
        for(var temp of uploadFiles) if(temp.name == file.name) {
            flag = true;
            break;
        }
        if(flag) return;

        // 加入文件列表
        uploadFiles.push(file);

        // 创建文件显示框
        var fileItem = document.createElement("div");
        fileItem.style.height = "100px";
        fileItem.style.width = "690px";
        fileItem.style.margin = "auto";
        fileItem.style.background = "white";
        fileItem.style.border = "1px solid lightgray";
        fileItem.style.borderRadius = "5px";
        fileItem.onmouseover = fileItem.onmouseup = function(){
            fileItem.style.border = "1px solid black";
        }
        fileItem.onmouseleave = function(){
            fileItem.style.border = "1px solid lightgray";
        }

        // 文件图标
        var fileIcon = document.createElement("img");
        fileIcon.style.width = "80px";
        fileIcon.style.height = "90%";
        fileIcon.style.display = "float";
        fileIcon.style.float = "left";
        fileIcon.style.position = "relative";
        fileIcon.style.top = "5px";
        fileIcon.style.left = "5px";
        // 图标判断
        if(file.type.split('/')[1] == "pdf") fileIcon.src = "../images/pdf.jpeg";
        else if(file.type.split('/')[1] == "exe") fileIcon.src = "../images/exe.jpeg";
        else if(file.type.split('/')[1] == "iso") fileIcon.src = "../images/iso.jpeg";
        else if(file.type.split('/')[1] == "dll") fileIcon.src = "../images/dll.jpeg";
        else if(file.type.split('/')[1] == "elf") fileIcon.src = "../images/elf.jpeg";
        else if(file.type.split('/')[1] == "rar") fileIcon.src = "../images/rar.jpeg";
        else if(file.type.split('/')[1] == "zip") fileIcon.src = "../images/zip.jpeg";
        else fileIcon.src = "../images/other.jpeg";

        // 文件名显示
        var fileName = document.createElement("span");
        fileName.innerText = file.name;
        fileName.style.display = "float";
        fileName.style.position = "relative";
        fileName.style.left = "10px";
        fileName.style.float = "left";
        fileName.style.width = "480px";
        fileName.style.height = "100%";
        fileName.style.lineHeight = "100px";

        // 文件删除按钮显示
        var deleteIcon = document.createElement("img");
        deleteIcon.src = "../images/delete.jpeg";
        deleteIcon.style.display = "float";
        deleteIcon.style.position = "relative";
        deleteIcon.style.float = "right";
        deleteIcon.style.right = "40px";
        deleteIcon.style.width = "20px";
        deleteIcon.style.height = "20px";
        deleteIcon.style.top = "40px";
        deleteIcon.onmouseover = function(){
            deleteIcon.style.border = "1px solid black";
        }
        deleteIcon.onmouseleave = function(){
            deleteIcon.style.border = "none";
        }
        deleteIcon.onmousedown = function(){
            deleteIcon.style.border = "3px solid black";
        }
        deleteIcon.onmouseup = function(){
            deleteIcon.style.border = "1px solid black";
        }
        // 设置删除
        deleteIcon.onclick = function(){
            uploadFiles.splice(uploadFiles.indexOf(file), 1);
            fileDisplay.removeChild(fileItem);
            // 如果全部删除恢复放大镜显示
            if(fileDisplay.children.length == 0) fileDisplay.appendChild(scanImage);
        }


        // 组建并显示文件显示框
        fileItem.appendChild(fileIcon);
        fileItem.appendChild(fileName);
        fileItem.appendChild(deleteIcon);
        fileDisplay.appendChild(fileItem);
    }
}

scanSubmitButton.onclick = function(){
    // 如果正在扫描，取消响应
    if(isScanning) return;

    // 检查状态
    if(scanSubmitButton.innerText == "scan") {
        // 如果没有文件，直接返回
        if(uploadFiles.length == 0) return;

        // 设置进入扫描状态
        isScanning = true;
        fileUploadButton.style.visibility = "hidden";
        scanSubmitButton.innerText = "scanning...";
        for(var temp of fileDisplay.children) {
            temp.children[2].src = "../images/scanning.gif";
            temp.children[2].onmouseleave = null;
            temp.children[2].onmouseover = null;
            temp.children[2].onmousedown = null;
            temp.children[2].onmouseup = null;
            temp.children[2].onclick = null;
        }

        // 逐个扫描并返回结果
        for(var i in uploadFiles) {
            var fileData = new FormData();
            fileData.append("ScanFile", uploadFiles[i]);

            // 提交文件
            fetch(window.parent.location.href + "scan", {method : "POST", body : fileData}).then(function(response){response.json().then(function(result){
                // 根据返回结果显示
                if(result.Total == 0) fileDisplay.children[i].children[2].src = "../images/correct.jpeg";
                else if(result.Total == 1) fileDisplay.children[i].children[2].src = "../images/wrong.jpeg";
                else fileDisplay.children[i].children[2].src = "../images/warning.jpg";

                // 储存结果
                scanResult.push(result);

                // 设置报告显示
                fileDisplay.children[i].onmousedown = function(){
                    fileDisplay.children[i].style.border = "2px solid black";
                }
                fileDisplay.children[i].onclick = function(){
                    // 申请显示报告
                    fetch(window.parent.location.href + "scanreport", {method : "POST", body : new FormData().append("Data", JSON.stringify(scanResult[i]))}).then(function(response){response.text().then(function(report){
                        window.open().document.write(report);
                    })});
                }
            })});
        }
        // 扫描结束设置
        reportNotice.style.visibility = "visible";
        scanSubmitButton.innerText = "back";
        isScanning = false;
    }
    else {
        // back重置
        uploadFiles = [];
        for(var temp of fileDisplay.children) fileDisplay.removeChild(temp);
        fileDisplay.appendChild(scanImage);
        fileUploadButton.style.visibility = "visible";
        reportNotice.style.visibility = "hidden";
        scanSubmitButton.innerText = "scan";
    }
    
}