'use strict'
var submitButton = document.getElementById("SearchSubmitButton");
var inputContent = document.getElementById("SearchSubmitContent");
var title = document.getElementById("Title");
var resultList = document.getElementById("ResultList");
var pageControl = document.getElementById("PageControl")
var allPage = document.getElementById("AllPage")
var nextPage = document.getElementById("NextPage")
var prePage = document.getElementById("PrePage")
window.searchState = false;
var page = 1
var searchCont = ""
var totalpage = 1

// 上一页与下一页按钮设置
nextPage.onclick = function(){
    if(page < totalpage) page += 1;
    submitButton.onclick();
}
prePage.onclick = function(){
    if(page > 1) page -= 1;
    submitButton.onclick();
}

submitButton.onclick  = function(){
    // 检查是否正在搜索
    if(searchState == true) return;
    searchState = true;
    if(searchCont != inputContent.value) {
        totalpage = 1
        page = 1;
    }
    searchCont = inputContent.value;
    // 检查空值
    if(inputContent.value == "") {
        // 淡化结果显示
        resultList.style.opacity = '1';
        var opacityTem = 1;
        pageControl.style.visibility = "hidden";
        var AEtimerResult = setInterval(function(){
            // 结果显示消失
            if(opacityTem <= 0) {
                resultList.style.visibility = "hidden";
                resultList.style.innerHTML = "";
                resultList.style.height = "auto";
                clearInterval(AEtimerResult);

                // 图标回移
                var startPosition = Number(title.style.getPropertyValue("margin").split("px")[0]);
                var endPosition = 160;
                var AEtimer = setInterval(function(){
                    if(startPosition >= endPosition - 10) {
                        title.style.setProperty("margin", Math.ceil(startPosition).toString() + "px 0px 50px 0px");
                        clearInterval(AEtimer);
                        searchState = false;
                        return;
                    }
                    startPosition += (endPosition - startPosition) / 100;
                    title.style.setProperty("margin", startPosition.toString() + "px 0px 50px 0px");
                })
                return;
            }
            opacityTem -= 0.01;
            resultList.style.opacity = opacityTem.toString();
        });
        return;
    }

    // 标记进入搜索状态
    submitButton.innerText = "搜索中...";

    // 向网站发送搜索请求，等待结果
    fetch(window.parent.location.href + "search?TimeSearch=" + inputContent.value + "&&Page=" + page).then(function(response){response.json().then(function(searchResult){
        // 标记非搜索状态
        submitButton.innerText = "搜索";

        // 重置输出结果
        resultList.innerHTML = "";

        // 加载页数信息
        totalpage = Number(searchResult['Pages'])
        if(totalpage <= 0) totalpage = 1
        allPage.innerText = page + ' / ' + totalpage.toString();

        // 加载显示项
        if(Object.keys(searchResult["Result"]).length == 0) {
            var resultItemTemplate = document.createElement("p");
            resultItemTemplate.style.fontSize = "30px";
            resultItemTemplate.style.fontWeight = "bold";
            resultItemTemplate.style.color = "white"
            resultItemTemplate.style.textAlign = "center"
            resultItemTemplate.innerText = "No result";
            resultList.appendChild(resultItemTemplate);
        }
		else{
			for(var item of searchResult["Result"]){
				var sha256 = document.createElement("div");
                sha256.style.height = "30px";
                sha256.style.width = "690px";
                sha256.style.margin = "auto";
				sha256.style.fontSize = "18px";
                sha256.style.color = "black"
				sha256.innerText = item;
                sha256.style.background = "white";
                sha256.style.border = "1px solid lightgray";
                sha256.style.textAlign = "center"
                sha256.style.borderRadius = "5px";
                sha256.style.opacity = "0.5"
                sha256.onclick = function(event){
                    event.srcElement.style.background = "gray";
                    window.open(window.parent.location.href + "/download/" + event.srcElement.innerText);
                }
                sha256.onmouseover = sha256.onmouseup = function(event){
                    event.srcElement.style.border = "1px solid black";
                    event.srcElement.style.background = "lightgray";
                }
                sha256.onmouseleave = function(event){
                    event.srcElement.style.border = "1px solid lightgray";
                    event.srcElement.style.background = "white";
                }
				resultList.appendChild(sha256);
			}

		}

        // 显示动画
        var startPosition = Number(title.style.getPropertyValue("margin").split("px")[0]);
        var endPosition = 30;
        // 图标上移
        var AEtimer = setInterval(function(){
            if(startPosition <= endPosition + 10) {
                title.style.setProperty("margin", Math.ceil(startPosition).toString() + "px 0px 50px 0px");
                clearInterval(AEtimer);

                // 显示结果
                resultList.style.visibility = "visible";
                resultList.style.opacity = '0';
                var opacityTemp = 0.1;
                var AEtimerResult = setInterval(function(){
                    if(opacityTemp >= 1) {
                        clearInterval(AEtimerResult);
                        searchState = false;
                        pageControl.style.visibility = "visible"
                        return;
                    }
                    // 透明度降低
                    opacityTemp += 0.01;
                    resultList.style.opacity = opacityTemp.toString();
                });
                return;
            }
            // 上移
            startPosition -= (startPosition - endPosition) / 100;
            title.style.setProperty("margin", startPosition.toString() + "px 0px 50px 0px");
        })})})
}


// 设置回车键发起搜索功能
document.onkeydown = function(event){
    if(event.keyCode == 13){
        submitButton.onclick();
    }
}